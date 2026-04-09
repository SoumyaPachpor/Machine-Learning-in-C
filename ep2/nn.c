#define NN_IMPLEMENTATION
#include "nn.h"

int main(void) {
    Mat x = mat_alloc(3, 3);
    Mat m = mat_alloc(3, 1);
    mat_fill(m, 3);
    Mat n = mat_alloc(1, 3);
    mat_fill(n, 3);
    mat_dot(x, m, n);
    mat_print(x);
    return 0;
}