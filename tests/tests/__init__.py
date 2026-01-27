"""Pytest-based end-to-end tests for TB2J.

This package integrates the shell-based TB2J_test_data scenarios under
`tests/data/tests` with pytest so they can be run and reported as
standard pytest tests.

Usage (from repository root):

    ./tests/init_test_data.sh  # ensure submodule is initialized
    pytest tests/tests -q

"""
