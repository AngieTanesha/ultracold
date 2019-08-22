#!/usr/bin/bash

# Exit on any error
set -eux

# Install texlive
texlive_setup() {
    su build -c "yay -Syu --noconfirm"
    su build -c "yay -S --noconfirm texlive-most biber otf-fira-code otf-eb-garamond pygmentize"
}

dir_setup() {
    mkdir -p LaTeX/images/tikz/
}

cleanup() {
    rm LaTeX/notes.pdf
}

main() {
    texlive_setup
    dir_setup
    cleanup
}

main
