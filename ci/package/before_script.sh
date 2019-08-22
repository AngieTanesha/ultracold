#!/usr/bin/bash

# Exit on any error
set -eux

# Install Python3
python_setup() {
    su build -c "yay -Syu --noconfirm"
    su build -c "yay -S --noconfirm python"
}


main() {
    python_setup
}

main
