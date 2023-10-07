#!/bin/bash

python3 -m coverage run --source=. --omit=main.py,tests.py tests.py
python3 -m coverage report -m 