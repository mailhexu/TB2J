# Test Data Submodule Setup

This directory contains scripts for managing the test data submodule for TB2J.

## Scripts Overview

### `setup_test_submodule.sh` (Run Once Locally)
**Purpose**: Adds the test data repository as a git submodule for the first time.

**When to use**: 
- Run this script once locally when you want to add the test data submodule to your repository
- Should be run by a maintainer/developer who has write access to the repository

**Usage:**
```bash
./setup_test_submodule.sh
```

**What it does:**
- Checks that the submodule doesn't already exist
- Runs `git submodule add https://github.com/mailhexu/TB2J_test_data.git tests/data`
- Creates the `.gitmodules` file and registers the submodule

**After running this script:**
```bash
git add .gitmodules tests/data
git commit -m "Add test data submodule"
git push
```

### `tests/init_test_data.sh` (Automatic in CI)
**Purpose**: Initializes existing submodules in CI/CD environments.

**When to use**: 
- Automatically run by GitHub Actions after checkout
- Can be run locally if you clone a repo that already has submodules defined

**Usage:**
```bash
./tests/init_test_data.sh
```

**What it does:**
- Checks if `.gitmodules` exists
- Runs `git submodule update --init --recursive` to initialize all submodules
- Verifies that the test data is properly checked out

## Workflow

1. **First-time setup** (maintainer runs locally):
   ```bash
   ./setup_test_submodule.sh
   git add .gitmodules tests/data
   git commit -m "Add test data submodule"
   git push
   ```

2. **CI/CD automatic initialization** (runs in GitHub Actions):
   ```bash
   git checkout --recurse-submodules  # Done by GitHub Actions
   ./tests/init_test_data.sh          # Safety check and verification
   ```

3. **Developers cloning the repo**:
   ```bash
   git clone --recurse-submodules https://github.com/mailhexu/TB2J.git
   # OR after regular clone:
   ./tests/init_test_data.sh
   ```

## Do You Need Both Scripts?

**Yes**, but they serve different purposes:

- **`setup_test_submodule.sh`**: One-time setup (like installation)
- **`init_test_data.sh`**: Runtime initialization (like starting a service)

## Directory Structure

After setup, your directory structure should look like:
```
TB2J/
├── setup_test_submodule.sh    # One-time submodule setup script (root level)
└── tests/
    ├── data/                  # Git submodule containing test data
    ├── init_test_data.sh      # CI initialization script
    ├── setup_test_data.sh     # Original test data setup script
    └── README.md              # This file
```

The scripts are organized as:
- **Root level**: `setup_test_submodule.sh` - for initial repository setup
- **tests/ directory**: `init_test_data.sh` - for runtime/CI initialization
