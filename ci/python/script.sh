#!/usr/bin/bash

# Exit on any error
set -eux

# Check formatting
check_format() {
    yapf --parallel --recursive --diff .
}

# Check lints
check_lint() {
    pylint --exit-zero *.py
}

# Run the code
check_run() {
    ./analytic_solution.py
    ./numerical_solution.py
}

main() {
    cd python
    export VENV_HOME_DIR=$(pipenv --venv)
    set +eu
    source $VENV_HOME_DIR/bin/activate
    set -eu

    check_format
    check_lint
    check_run
}

main
