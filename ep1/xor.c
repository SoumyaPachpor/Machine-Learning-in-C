#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

typedef struct
{
    float or_w1;
    float or_w2;
    float or_b;

    float and_w1;
    float and_w2;
    float and_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
} Xor;
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
sample xor_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
};

sample* train_data = or_train;

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float forward(Xor m, float x1, float x2)
{
    // layer 1
    float a = sigmoidf(m.or_b + x1 * m.or_w1 + x2 * m.or_w2);
    float b = sigmoidf(m.nand_b + x1 * m.nand_w1 + x2 * m.nand_w2);
    // layer 2
    float c = sigmoidf(m.and_b + a * m.and_w1 + b * m.and_w2);
    return c;
}
#define train_size sizeof(xor_train) / sizeof(xor_train[0])

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float cost(Xor m)
{
    float result = 0.0;
    for (size_t i = 0; i < train_size; i++)
    {
        float x1 = train_data[i][0];
        float x2 = train_data[i][1];
        float y = forward(m, x1, x2);
        float d = y - train_data[i][2];
        result += d * d;
    }
    return result /= train_size;
}

void test(Xor m)
{
    for (size_t i = 0; i <= 1; i++)
    {
        for (size_t j = 0; j <= 1; j++)
        {
            printf("%lu %lu = %f\n",
                   (unsigned long)i,
                   (unsigned long)j,
                   forward(m, i, j));
        }
    }
}

void internal_test(float w1, float w2, float b, char ch)
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
    printf("\n");
}

Xor rand_xor()
{
    Xor m;
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();
    return m;
}

void print_xor(Xor m)
{
    printf("or_w1= %f\n", m.or_w1);
    printf("or_w2= %f\n", m.or_w2);
    printf("or_b= %f\n", m.or_b);

    printf("and_w1= %f\n", m.and_w1);
    printf("and_w2= %f\n", m.and_w2);
    printf("and_b= %f\n", m.and_b);

    printf("nand_w1= %f\n", m.nand_w1);
    printf("nand_w2= %f\n", m.nand_w2);
    printf("nand_b= %f\n", m.nand_b);
}

Xor finite_diff(Xor m, float eps)
{
    Xor g;

    float c = cost(m);
    float saved;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    return g;
}

Xor learn(Xor m, Xor g, float rate)
{
    m.or_w1 -= rate * g.or_w1;
    m.or_w2 -= rate * g.or_w2;
    m.or_b -= rate * g.or_b;

    m.and_w1 -= rate * g.and_w1;
    m.and_w2 -= rate * g.and_w2;
    m.and_b -= rate * g.and_b;

    m.nand_w1 -= rate * g.nand_w1;
    m.nand_w2 -= rate * g.nand_w2;
    m.nand_b -= rate * g.nand_b;
    return m;
}

int main(void)
{
    srand(time(0));
    Xor m = rand_xor();
    float eps = 1e-2;
    float rate = 1e-0;
    // print_xor(m);
    for (size_t i = 0; i < 100 * 1000; i++)
    {
        Xor g = finite_diff(m, eps);
        m = learn(m, g, rate);
        printf("cost: %f\n", cost(m));
    }
    printf("cost: %f\n", cost(m));
    test(m);
    internal_test(m.and_w1, m.and_w2, m.and_b, '&');
    internal_test(m.or_w1, m.or_w2, m.or_b, '|');
    internal_test(m.nand_w1, m.nand_w2, m.nand_b, 'N');
    return 0;
}