#!/usr/bin/env bash
rm ./dist/*
python3.11 setup.py sdist bdist_wheel
python3.11 -m twine upload --repository pypi dist/* --verbose
