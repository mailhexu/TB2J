#!/usr/bin/env bash
rm -rf ./dist/*
#python3 setup.py sdist bdist_wheel
#python -m build
uv build
python -m twine upload --repository pypi dist/* --verbose
