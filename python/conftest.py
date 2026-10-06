"""The suite runs against the installed package, not the source tree.

`make test-py` builds the extension and installs the project into the
virtualenv before pytest starts. Importing `python/lodestar` directly would
test files that are not in the wheel — and would fail anyway, since the
compiled submodule only exists once maturin has put it there.

Importing the package here turns a skipped build into one obvious failure at
collection time instead of twenty confusing ones.
"""

import lodestar  # noqa: F401
