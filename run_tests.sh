#!/bin/bash
export PYTHONPATH=$(pwd)
pip install coverage
coverage run -m pytest tests
coverage report

