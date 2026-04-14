#!/bin/bash

set +xe

# gcc -Wall -Werror simple.c -o simple && ./simple.exe
# gcc -Wall -Werror gates.c -o gates -lm && ./gates.exe
gcc -Wall -Werror xor.c -o xor -lm && ./xor.exe