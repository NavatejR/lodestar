//! Fuzz the metadata filter language.
//!
//! The expression text arrives straight from HTTP query parameters and the
//! Python bindings, so `parse` sees strings no application would produce.
//! The parser must either reject the input or produce an expression that
//! evaluates without panicking against well-formed, malformed and absent
//! metadata alike.

#![no_main]

use libfuzzer_sys::fuzz_target;
use lodestar_ann_index::filter;

fuzz_target!(|data: &[u8]| {
    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    // One row with every value shape, one sparse row, one vector that has no
    // metadata at all: the three cases `compile` and `matches` must agree on.
    let mut full = filter::Metadata::new();
    full.insert("k".into(), filter::Value::Str(input.to_string()));
    full.insert("n".into(), filter::Value::Num(0.5));
    full.insert("b".into(), filter::Value::Bool(true));
    full.insert(
        "l".into(),
        filter::Value::List(vec![filter::Value::Str("x".into())]),
    );
    let mut sparse = filter::Metadata::new();
    sparse.insert("n".into(), filter::Value::Num(-1.0));
    let table = [Some(full.clone()), Some(sparse), None];

    if let Ok(expr) = filter::parse(input) {
        let _ = expr.matches(table[0].as_ref());
        let _ = expr.matches(None);
        let _ = filter::compile(&expr, &table);
    }
    let _ = filter::parse_and_compile(input, &table);
});
