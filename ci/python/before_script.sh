#!/usr/bin/bash

# Exit on any error
set -eux

# Install rustup
python_setup() {
    su build -c "yay -Syu --noconfirm"
    su build -c "yay -S python-pipenv python-pylint yapf --noconfirm"
}

pipenv_setup() {
    cd python
    pipenv install
}

main() {
    python_setup
    pipenv_setup
}

main
