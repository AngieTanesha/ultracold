#!/usr/bin/bash

# Exit on any error
set -eux

# Run the code
compile_tex() {
    latexmk notes.tex
}

main() {
    cd LaTeX
    compile_tex
}

main
