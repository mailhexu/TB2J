# TB2J Test Suite

This directory contains the TB2J test suite, organized into unit and end-to-end (E2E) tiers, plus a git submodule for test data.

## Directory Structure

```
tests/
├── unit/                  # Fast unit tests (no external data needed)
│   ├── test_bruno_correction.py
│   ├── test_bruno_realspace.py
│   ├── test_cli_remove_sublattice.py
│   └── test_cli_toggle_exchange.py
├── e2e/                   # End-to-end tests (need test data submodule)
│   ├── test_e2e_tb2j.py
│   ├── test_qspace_vs_realspace.py
│   └── test_qspace_ncl_vs_realspace.py
├── data/                  # Git submodule (TB2J_test_data)
├── init_test_data.sh      # CI: initialize submodule
├── update_test_data.sh    # Update submodule to latest
├── conftest.py            # Shared pytest fixtures
└── README.md              # This file
```

## Running Tests

### Prerequisites

1. Install Python dependencies (from the repository root):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install .
   python -m pip install pytest
   ```
2. For E2E tests, initialize the test data submodule:
   ```bash
   ./tests/init_test_data.sh
   ```

### Quick Commands

```bash
# All tests
pytest

# Unit tests only (fast, no data needed)
pytest tests/unit/ -v
pytest -m unit -v

# E2E tests only (needs submodule)
pytest tests/e2e/ -v
pytest -m e2e -v

# Skip E2E tests
pytest -m "not e2e" -v
```

## Test Data Submodule

Test data lives in a separate repository: `https://github.com/mailhexu/TB2J_test_data.git`

### First-time setup (maintainer):
```bash
git submodule add https://github.com/mailhexu/TB2J_test_data.git tests/data
git add .gitmodules tests/data
git commit -m "Add test data submodule"
```

### Developer clone:
```bash
git clone --recurse-submodules https://github.com/mailhexu/TB2J.git
# OR after regular clone:
./tests/init_test_data.sh
```

### Update test data:
```bash
./tests/update_test_data.sh
git add tests/data
git commit -m "Update tests/data submodule"
```

## Adding New E2E Tests

See `tests/data/README_add_test.md` for instructions on adding new test scenarios using the `create_new_test.py` utility.
