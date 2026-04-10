#!/bin/bash

set +xe

# gcc -Wall -Werror nn.c -o nn -lm
# gcc -std=c99 -Wall -Werror xor.c -o xor -lm
gcc -std=c99 -Wall -Werror layers.c -o layers -lm