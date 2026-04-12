#!/bin/bash

set +xe
# -std=c99 is needed for %zu in printf, which is used in nn_print
# gcc -Wall -Werror nn.c -o nn -lm  && ./nn.exe
# gcc -std=c99 -Wall -Werror xor.c -o xor -lm && ./xor.exe
# gcc -std=c99 -Wall -Werror layers.c -o layers -lm && ./layers.exe
gcc -std=c99 -Wall -Werror adder.c -o adder -lm && ./adder.exe