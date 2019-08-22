#!/usr/bin/bash

# Exit on any error
set -eux

package() {
    ./package-for-students
}

main() {
    package
}

main
