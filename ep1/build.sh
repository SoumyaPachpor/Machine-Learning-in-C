#!/bin/bash

set +xe

# gcc -Wall -Werror simple.c -o simple
# gcc -Wall -Werror gates.c -o gates -lm
gcc -Wall -Werror xor.c -o xor -lm