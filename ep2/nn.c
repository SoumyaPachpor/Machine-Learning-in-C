#define NN_IMPLEMENTATION
#include "nn.h"

int main(void) {
    Mat m = mat_alloc(3, 4);
    mat_rand(m, 3, 4);
    mat_print(m);
    return 0;
}