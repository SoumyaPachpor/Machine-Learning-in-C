#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

typedef float sample[3];

sample and_train[] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};
sample or_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};
sample nand_train[] = {
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample* train_data = nand_train;
#define train_size sizeof(nand_train) / sizeof(nand_train[0])

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float cost(float w1, float w2, float b)
{
    float result = 0.0;
    for (size_t i = 0; i < train_size; i++)
    {
        float x1 = train_data[i][0];
        float x2 = train_data[i][1];
        float y = sigmoidf(w1 * x1 + w2 * x2 + b);
        float d = y - train_data[i][2];
        result += d * d;
    }
    return result /= train_size;
}

void test(float w1, float w2, float b, char ch)
{
    for (size_t i = 0; i <= 1; i++)
    {
        for (size_t j = 0; j <= 1; j++)
        {
            printf("%lu %c %lu = %f\n",
                   (unsigned long)i,
                   ch,
                   (unsigned long)j,
                   sigmoidf(w1 * i + w2 * j + b));
        }
    }
}

int main(void)
{
    srand(time(0));
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float() * 5.0f;
    float eps = 1e-1;
    float r = 1e-1;
    for (size_t i = 0; i < 500*1000; i++)
    {
        float c = cost(w1, w2, b);
        float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        float db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= r * dw1;
        w2 -= r * dw2;
        b -= r * db;
        // printf("cost: %f, w1: %f, w2: %f, b: %f\n", c, w1, w2, b);
    }
    printf("cost: %f, w1: %f, w2: %f, b: %f\n", cost(w1, w2, b), w1, w2, b);
    test(w1, w2, b, '&');
    return 0;
}