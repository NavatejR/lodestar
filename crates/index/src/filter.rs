//! Metadata filtering.
//!
//! Vector search without metadata filtering is rarely useful: "find similar
//! documents, but only ones this user may read", "only products in stock",
//! "only records from this tenant". Lodestar supports it with a small
//! expression language:
//!
//! ```text
//! category == "news" AND score >= 0.8
//! tenant IN ("acme", "globex") AND NOT archived
//! tags CONTAINS "rust"
//! author EXISTS
//! ```
//!
//! ## Why a bitset
//!
//! Evaluating an expression per visited node during traversal would add a
//! string lookup and a match to the innermost loop of the hottest code path.
//! Instead the expression is evaluated once against the whole collection,
//! producing a [`BitSet`]; traversal then costs one bit test per node. For a
//! selective filter that is the difference between "usable" and "unusable".
//!
//! ## Missing fields
//!
//! Any comparison against a field that a vector does not define evaluates to
//! `false` — including `!=`. This is the semantics every filtered search system
//! converges on, because the alternative ("missing != x is true") makes an
//! exclusion filter silently match everything that has no metadata at all.
//! Use `NOT field EXISTS` to select vectors with no value.

use std::collections::HashMap;

use lodestar_ann_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// A metadata value. Deliberately JSON-shaped, since that is what arrives over
/// the HTTP API and out of the Python bindings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// Boolean.
    Bool(bool),
    /// Number (always stored as `f64`; vector metadata is not used for
    /// arithmetic-heavy work).
    Num(f64),
    /// String.
    Str(String),
    /// List of values, matched by `IN` and `CONTAINS`.
    List(Vec<Value>),
}

impl Value {
    /// Renders the value for error messages and `Display`-based comparisons.
    fn as_comparable_string(&self) -> String {
        match self {
            Self::Bool(value) => value.to_string(),
            Self::Num(value) => {
                // Render integral floats without a trailing `.0` so that a
                // metadata value of `3` written as JSON compares equal to the
                // string "3".
                if value.fract() == 0.0 && value.abs() < 1e15 {
                    format!("{}", *value as i64)
                } else {
                    value.to_string()
                }
            }
            Self::Str(value) => value.clone(),
            Self::List(values) => values
                .iter()
                .map(Self::as_comparable_string)
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Num(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::Str(value.to_string())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

/// Metadata attached to one vector.
pub type Metadata = HashMap<String, Value>;

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl Op {
    /// The operator as it is written in the query language.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A parsed filter expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Both sides must match.
    And(Box<Self>, Box<Self>),
    /// Either side may match.
    Or(Box<Self>, Box<Self>),
    /// Inverts the inner expression.
    Not(Box<Self>),
    /// `field OP value`
    Compare {
        /// Metadata key.
        field: String,
        /// Comparison operator.
        op: Op,
        /// Right-hand side.
        value: Value,
    },
    /// `field IN (a, b, c)`
    In {
        /// Metadata key.
        field: String,
        /// Candidate values.
        values: Vec<Value>,
    },
    /// `field CONTAINS value` — substring for strings, membership for lists.
    Contains {
        /// Metadata key.
        field: String,
        /// Value to look for.
        value: Value,
    },
    /// `field EXISTS`
    Exists {
        /// Metadata key.
        field: String,
    },
}

impl Expr {
    /// Matches `metadata` against this expression.
    ///
    /// `None` metadata never matches a comparison, but does match
    /// `NOT (field EXISTS)`.
    #[must_use]
    pub fn matches(&self, metadata: Option<&Metadata>) -> bool {
        match self {
            Self::And(left, right) => left.matches(metadata) && right.matches(metadata),
            Self::Or(left, right) => left.matches(metadata) || right.matches(metadata),
            Self::Not(inner) => !inner.matches(metadata),
            Self::Exists { field } => metadata.is_some_and(|map| map.contains_key(field)),
            Self::Compare { field, op, value } => {
                let Some(found) = metadata.and_then(|map| map.get(field)) else {
                    return false;
                };
                compare(found, *op, value)
            }
            Self::In { field, values } => {
                let Some(found) = metadata.and_then(|map| map.get(field)) else {
                    return false;
                };
                values
                    .iter()
                    .any(|candidate| compare(found, Op::Eq, candidate))
            }
            Self::Contains { field, value } => {
                let Some(found) = metadata.and_then(|map| map.get(field)) else {
                    return false;
                };
                match found {
                    Value::List(items) => items.iter().any(|item| compare(item, Op::Eq, value)),
                    Value::Str(text) => match value {
                        Value::Str(needle) => text.contains(needle.as_str()),
                        other => text.contains(&other.as_comparable_string()),
                    },
                    _ => false,
                }
            }
        }
    }

    /// Collects every field name referenced by the expression.
    #[must_use]
    pub fn fields(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_fields(&mut out);
        out
    }

    fn collect_fields<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::And(left, right) | Self::Or(left, right) => {
                left.collect_fields(out);
                right.collect_fields(out);
            }
            Self::Not(inner) => inner.collect_fields(out),
            Self::Compare { field, .. }
            | Self::In { field, .. }
            | Self::Contains { field, .. }
            | Self::Exists { field } => out.push(field),
        }
    }
}

/// Compares a metadata value against a literal.
///
/// Ordering comparisons (`<`, `<=`, `>`, `>=`) are defined within a type:
/// numerically for numbers, lexicographically (by Unicode scalar value) for
/// strings, which is what makes `published > "2024-01-01"` work on ISO dates.
/// Across types an ordering comparison would need a collation nobody agrees on,
/// so it is simply false rather than an error.
fn compare(found: &Value, op: Op, literal: &Value) -> bool {
    match (found, literal) {
        (Value::Num(a), Value::Num(b)) => match op {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        },
        (Value::Str(a), Value::Str(b)) => match op {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        },
        (Value::Bool(a), Value::Bool(b)) => match op {
            Op::Eq => a == b,
            Op::Ne => a != b,
            _ => false,
        },
        // Cross-type equality is decided by rendering, which is what makes
        // `count == "3"` behave the way a JSON client expects.
        (a, b) if matches!(op, Op::Eq | Op::Ne) => {
            let equal = a.as_comparable_string() == b.as_comparable_string();
            if op == Op::Eq { equal } else { !equal }
        }
        _ => false,
    }
}

/// A compact set of node indices.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Creates an empty set able to index `len` nodes.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// Number of addressable nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the set addresses no nodes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Marks `index` as present. Out-of-range indices are ignored.
    pub fn set(&mut self, index: usize) {
        if index < self.len {
            self.words[index / 64] |= 1u64 << (index % 64);
        }
    }

    /// Tests `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        index < self.len && (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }

    /// Number of present indices.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Bytes held by the bitset.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    /// Iterates the present indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, word)| {
                let mut bits = *word;
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some(word_index * 64 + bit)
                })
            })
    }
}

/// Evaluates `expr` once per vector and returns the matching node set.
#[must_use]
pub fn compile(expr: &Expr, metadata: &[Option<Metadata>]) -> BitSet {
    let mut bits = BitSet::new(metadata.len());
    for (index, entry) in metadata.iter().enumerate() {
        if expr.matches(entry.as_ref()) {
            bits.set(index);
        }
    }
    bits
}

/// Parses a filter expression.
///
/// # Errors
///
/// Returns [`Error::InvalidParameter`] naming the offending token when the
/// expression cannot be parsed.
pub fn parse(input: &str) -> Result<Expr> {
    let tokens = tokenize(input)?;
    let mut parser = Parser {
        tokens,
        position: 0,
    };
    let expr = parser.parse_or()?;
    if !matches!(parser.peek(), Token::End) {
        return Err(parser.error(&format!("unexpected token {:?}", parser.peek())));
    }
    Ok(expr)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Num(f64),
    Bool(bool),
    And,
    Or,
    Not,
    In,
    Contains,
    Exists,
    Op(Op),
    LParen,
    RParen,
    Comma,
    End,
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '=' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Op(Op::Eq));
                    i += 2;
                } else {
                    return Err(Error::InvalidParameter {
                        name: "filter",
                        reason: format!("expected `==` at position {i}, found `=`"),
                    });
                }
            }
            '!' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Op(Op::Ne));
                    i += 2;
                } else {
                    return Err(Error::InvalidParameter {
                        name: "filter",
                        reason: format!("expected `!=` at position {i}, found `!`"),
                    });
                }
            }
            '<' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Op(Op::Le));
                    i += 2;
                } else {
                    tokens.push(Token::Op(Op::Lt));
                    i += 1;
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Op(Op::Ge));
                    i += 2;
                } else {
                    tokens.push(Token::Op(Op::Gt));
                    i += 1;
                }
            }
            '"' | '\'' => {
                let quote = c;
                let mut text = String::new();
                i += 1;
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch == '\\' {
                        if let Some(next) = chars.get(i + 1) {
                            text.push(*next);
                            i += 2;
                            continue;
                        }
                    }
                    if ch == quote {
                        closed = true;
                        i += 1;
                        break;
                    }
                    text.push(ch);
                    i += 1;
                }
                if !closed {
                    return Err(Error::InvalidParameter {
                        name: "filter",
                        reason: "unterminated string literal".to_string(),
                    });
                }
                tokens.push(Token::Str(text));
            }
            c if c.is_ascii_digit()
                || (c == '-' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) =>
            {
                let start = i;
                i += 1;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || chars[i] == '-')
                {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                let value = text.parse::<f64>().map_err(|_| Error::InvalidParameter {
                    name: "filter",
                    reason: format!("`{text}` is not a valid number"),
                })?;
                tokens.push(Token::Num(value));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_alphanumeric()
                        || chars[i] == '_'
                        || chars[i] == '.'
                        || chars[i] == '-')
                {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                match text.to_ascii_uppercase().as_str() {
                    "AND" => tokens.push(Token::And),
                    "OR" => tokens.push(Token::Or),
                    "NOT" => tokens.push(Token::Not),
                    "IN" => tokens.push(Token::In),
                    "CONTAINS" => tokens.push(Token::Contains),
                    "EXISTS" => tokens.push(Token::Exists),
                    "TRUE" => tokens.push(Token::Bool(true)),
                    "FALSE" => tokens.push(Token::Bool(false)),
                    _ => tokens.push(Token::Ident(text)),
                }
            }
            other => {
                return Err(Error::InvalidParameter {
                    name: "filter",
                    reason: format!("unexpected character `{other}` at position {i}"),
                });
            }
        }
    }
    tokens.push(Token::End);
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::End)
    }

    fn next(&mut self) -> Token {
        let token = self.peek().clone();
        self.position += 1;
        token
    }

    fn error(&self, detail: &str) -> Error {
        Error::InvalidParameter {
            name: "filter",
            reason: detail.to_string(),
        }
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::Or) {
            self.next();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek(), Token::And) {
            self.next();
            let right = self.parse_unary()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        if matches!(self.peek(), Token::Not) {
            self.next();
            return Ok(Expr::Not(Box::new(self.parse_unary()?)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        if matches!(self.peek(), Token::LParen) {
            self.next();
            let inner = self.parse_or()?;
            if !matches!(self.peek(), Token::RParen) {
                return Err(self.error("expected `)`"));
            }
            self.next();
            return Ok(inner);
        }

        let Token::Ident(field) = self.next() else {
            return Err(self.error(&format!("expected a field name, found {:?}", self.peek())));
        };

        match self.next() {
            Token::Op(op) => {
                let value = self.parse_literal()?;
                Ok(Expr::Compare { field, op, value })
            }
            Token::In => {
                if !matches!(self.peek(), Token::LParen) {
                    return Err(self.error("expected `(` after IN"));
                }
                self.next();
                let mut values = Vec::new();
                loop {
                    values.push(self.parse_literal()?);
                    match self.peek() {
                        Token::Comma => {
                            self.next();
                        }
                        Token::RParen => {
                            self.next();
                            break;
                        }
                        other => {
                            return Err(
                                self.error(&format!("expected `,` or `)`, found {other:?}"))
                            );
                        }
                    }
                }
                if values.is_empty() {
                    return Err(self.error("IN requires at least one value"));
                }
                Ok(Expr::In { field, values })
            }
            Token::Contains => {
                let value = self.parse_literal()?;
                Ok(Expr::Contains { field, value })
            }
            Token::Exists => Ok(Expr::Exists { field }),
            other => Err(self.error(&format!(
                "expected an operator after `{field}`, found {other:?}"
            ))),
        }
    }

    fn parse_literal(&mut self) -> Result<Value> {
        match self.next() {
            Token::Str(text) => Ok(Value::Str(text)),
            Token::Num(value) => Ok(Value::Num(value)),
            Token::Bool(value) => Ok(Value::Bool(value)),
            Token::Ident(name) => Ok(Value::Str(name)),
            other => Err(self.error(&format!("expected a value, found {other:?}"))),
        }
    }
}

/// Convenience: parse and compile in one step.
///
/// # Errors
///
/// As [`parse`].
pub fn parse_and_compile(input: &str, metadata: &[Option<Metadata>]) -> Result<BitSet> {
    Ok(compile(&parse(input)?, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(pairs: &[(&str, Value)]) -> Metadata {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    fn sample() -> Vec<Option<Metadata>> {
        vec![
            Some(metadata(&[
                ("category", "news".into()),
                ("score", 0.9f64.into()),
                ("views", 120f64.into()),
                ("archived", false.into()),
                ("tags", Value::List(vec!["rust".into(), "search".into()])),
            ])),
            Some(metadata(&[
                ("category", "blog".into()),
                ("score", 0.4f64.into()),
                ("views", 10f64.into()),
                ("archived", true.into()),
                ("tags", Value::List(vec!["python".into()])),
            ])),
            None,
            Some(metadata(&[
                ("category", "news".into()),
                ("score", 0.75f64.into()),
            ])),
        ]
    }

    fn matches(expr: &str, index: usize) -> bool {
        let data = sample();
        let parsed = parse(expr).unwrap();
        parsed.matches(data[index].as_ref())
    }

    #[test]
    fn equality_matches_only_the_right_rows() {
        assert!(matches("category == \"news\"", 0));
        assert!(!matches("category == \"news\"", 1));
        // Bare identifiers are treated as strings, so quoting is optional.
        assert!(matches("category == news", 0));
    }

    #[test]
    fn missing_metadata_never_matches_a_comparison() {
        assert!(!matches("category == \"news\"", 2));
        assert!(!matches("category != \"news\"", 2));
        assert!(!matches("score >= 0", 2));
    }

    #[test]
    fn not_exists_selects_rows_without_metadata() {
        assert!(matches("NOT category EXISTS", 2));
        assert!(!matches("NOT category EXISTS", 0));
        assert!(matches("category EXISTS", 0));
    }

    #[test]
    fn boolean_fields_compare_correctly() {
        assert!(matches("archived == true", 1));
        assert!(!matches("archived == true", 0));
        assert!(matches("archived == false", 0));
    }

    #[test]
    fn numeric_ordering_works() {
        assert!(matches("score >= 0.8", 0));
        assert!(!matches("score >= 0.8", 1));
        assert!(matches("views < 50", 1));
        assert!(matches("views <= 10", 1));
        assert!(matches("views > 100", 0));
    }

    #[test]
    fn ordering_compares_within_a_type_and_is_false_across_types() {
        // Same type: lexicographic on strings, where "news" sorts after "a".
        assert!(matches("category > \"a\"", 0));
        assert!(!matches("category < \"a\"", 0));
        // Cross type: false rather than an error, so one malformed clause in a
        // generated query cannot fail the whole search.
        assert!(!matches("category > 5", 0));
        assert!(!matches("score > \"1\"", 0));
    }

    #[test]
    fn and_or_not_combine() {
        assert!(matches("category == news AND score > 0.8", 0));
        assert!(!matches("category == news AND score > 0.8", 3));
        assert!(matches("category == blog OR score > 0.8", 1));
        assert!(matches("NOT category == news", 1));
        assert!(matches(
            "(category == news OR category == blog) AND archived == false",
            0
        ));
    }

    #[test]
    fn in_lists_match_any_member() {
        assert!(matches("category IN (news, blog)", 0));
        assert!(matches("category IN (\"news\", \"blog\")", 1));
        assert!(!matches("category IN (blog)", 0));
    }

    #[test]
    fn contains_handles_strings_and_lists() {
        assert!(matches("tags CONTAINS rust", 0));
        assert!(!matches("tags CONTAINS rust", 1));
        assert!(matches("category CONTAINS new", 0));
        assert!(!matches("category CONTAINS new", 1));
    }

    #[test]
    fn empty_in_list_is_rejected() {
        assert!(parse("category IN ()").is_err());
    }

    #[test]
    fn parse_errors_name_the_problem() {
        for bad in [
            "category ==",
            "category = news",
            "(category == news",
            "category == news extra",
            "score > ",
            "category ! news",
        ] {
            assert!(parse(bad).is_err(), "expected `{bad}` to fail");
        }
    }

    #[test]
    fn unterminated_strings_are_rejected() {
        let err = parse("category == \"news").unwrap_err();
        assert!(err.to_string().contains("unterminated"));
    }

    #[test]
    fn escapes_inside_strings_are_honoured() {
        let expr = parse("name == \"a\\\"b\"").unwrap();
        assert_eq!(
            expr,
            Expr::Compare {
                field: "name".to_string(),
                op: Op::Eq,
                value: Value::Str("a\"b".to_string()),
            }
        );
    }

    #[test]
    fn case_insensitive_keywords() {
        assert!(parse("a == 1 and b == 2 or c == 3").is_ok());
        assert!(parse("a == 1 AND b == 2").is_ok());
        assert!(parse("tags contains rust").is_ok());
    }

    #[test]
    fn fields_are_reported() {
        let expr = parse("a == 1 AND (b == 2 OR c EXISTS)").unwrap();
        let mut fields = expr.fields();
        fields.sort_unstable();
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn bitset_compilation_matches_row_by_row_evaluation() {
        let data = sample();
        let expr = parse("score >= 0.7").unwrap();
        let bits = compile(&expr, &data);
        assert_eq!(bits.count_ones(), 2);
        assert!(bits.get(0));
        assert!(!bits.get(1));
        assert!(!bits.get(2));
        assert!(bits.get(3));
        for (index, entry) in data.iter().enumerate() {
            assert_eq!(bits.get(index), expr.matches(entry.as_ref()));
        }
    }

    #[test]
    fn bitset_iteration_visits_present_indices() {
        let mut bits = BitSet::new(200);
        for index in [0usize, 63, 64, 65, 199] {
            bits.set(index);
        }
        let mut found: Vec<usize> = bits.iter().collect();
        found.sort_unstable();
        assert_eq!(found, vec![0, 63, 64, 65, 199]);
        assert_eq!(bits.count_ones(), 5);
        // Out-of-range writes and reads are ignored rather than panicking.
        bits.set(500);
        assert!(!bits.get(500));
        assert_eq!(bits.count_ones(), 5);
        assert!(bits.memory_bytes() >= 200 / 8);
    }

    #[test]
    fn parse_and_compile_is_equivalent_to_the_two_step_path() {
        let data = sample();
        let direct = parse_and_compile("archived == false OR views > 100", &data).unwrap();
        let manual = compile(&parse("archived == false OR views > 100").unwrap(), &data);
        assert_eq!(direct, manual);
    }

    #[test]
    fn cross_type_equality_renders_consistently() {
        // `count` is a number in the metadata but the filter uses a string.
        let row = metadata(&[("count", 3f64.into())]);
        let expr = parse("count == \"3\"").unwrap();
        assert!(expr.matches(Some(&row)));
        let expr = parse("count != \"4\"").unwrap();
        assert!(expr.matches(Some(&row)));
    }
}
