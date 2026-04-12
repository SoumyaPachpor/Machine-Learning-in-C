#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>
#include <stdio.h>

float td[] = {
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 1,
    1, 1, 0, 0, 1,
    1, 1, 1, 1, 1
};

int main(void)
{
    srand(time(0));

    size_t stride = 5; // FIXED
    size_t n = sizeof(td) / sizeof(td[0]) / stride;

    Mat ti = {
        .rows = n,
        .cols = 3,
        .stride = stride,
        .es = td,
    };

    Mat to = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td + 3, // FIXED (outputs start at index 3)
    };

    size_t arch[] = {3, 6, 2}; // slightly bigger = easier learning
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, 0, 1);

    float eps = 1e-2;
    float rate = 0.1; // FIXED (1 is too aggressive)

    for (size_t i = 0; i < 20000; i++)
    {
        nn_fill(g, 0); // IMPORTANT
        nn_finite_diff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);

        if (i % 2000 == 0) {
            printf("cost = %f\n", nn_cost(nn, ti, to));
        }
    }

    printf("\nTesting:\n");

    for (size_t a = 0; a < 2; a++)
    {
        for (size_t b = 0; b < 2; b++)
        {
            for (size_t cin = 0; cin < 2; cin++)
            {
                MAT_AT(NN_INPUT(nn), 0, 0) = a;
                MAT_AT(NN_INPUT(nn), 0, 1) = b;
                MAT_AT(NN_INPUT(nn), 0, 2) = cin;

                nn_forward(nn);

                float sum   = MAT_AT(NN_OUTPUT(nn), 0, 0);
                float carry = MAT_AT(NN_OUTPUT(nn), 0, 1);

                printf("%zu %zu %zu -> sum=%.3f carry=%.3f\n",
                       a, b, cin, sum, carry);
            }
        }
    }

    return 0;
}