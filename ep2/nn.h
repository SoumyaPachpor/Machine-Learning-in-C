#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#define ARRAY_LEN(xs) sizeof(xs) / sizeof(xs[0])

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) ((m).es[(i) * (m).stride + (j)])
#define MAT_PRINT(m) mat_print(m, #m, 0)

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
void mat_print(Mat m, const char *name, int padding);

float rand_float();
float sigmoidf(float x);

typedef struct
{
    size_t count;
    Mat *ws; // weights
    Mat *bs; // biases
    Mat *as; // the amount of activations will be count + 1
} NN;
#define NN_PRINT(m) nn_print(m, #m);
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
void nn_fill(NN nn, float val);
void nn_forward(NN nn);
void nn_finite_diff(NN m, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
float nn_cost(NN nn, Mat ti, Mat to);
#endif // NN_H_

#ifdef NN_IMPLEMENTATION

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = (void *)NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);
    for (size_t i = 0; i < a.rows; i++)
    {
        for (size_t j = 0; j < b.cols; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < a.cols; k++)
            {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(dst, i, j) = sum;
        }
    }
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_sum(Mat dst, Mat m)
{
    NN_ASSERT(dst.rows == m.rows);
    NN_ASSERT(dst.cols == m.cols);

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}
Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)};
}

void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_print(Mat m, const char *name, int padding)
{
    printf("%*s%s = [\n", padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    ", padding, "", name);
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", padding, "");
}

// size_t arch[] = {2, 2, 1};
// NN nn = nn_alloc(arch, ARRAY_LEN(arch));

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1; // as the first layer is the input layer, it doesn't have weights and biases

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = {\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buf, sizeof(buf), "ws%lu", (unsigned long)i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%lu", (unsigned long)i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("}\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_fill(NN nn, float val)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_fill(nn.ws[i], val);
        mat_fill(nn.bs[i], val);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;

    float c = 0.0;
    for (size_t i = 0; i < n; i++)
    {

        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            // som: d = predicted - expected
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    return c / n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float c = nn_cost(nn, ti, to);
    for (size_t k = 0; k < nn.count; k++)
    {
        for (size_t i = 0; i < nn.ws[k].rows; i++)
        {
            for (size_t j = 0; j < nn.ws[k].cols; j++)
            {
                float saved = MAT_AT(nn.ws[k], i, j);
                MAT_AT(nn.ws[k], i, j) += eps;
                MAT_AT(g.ws[k], i, j) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.ws[k], i, j) = saved;
            }
        }
        for (size_t i = 0; i < nn.bs[k].rows; i++)
        {
            for (size_t j = 0; j < nn.bs[k].cols; j++)
            {
                float saved = MAT_AT(nn.bs[k], i, j);
                MAT_AT(nn.bs[k], i, j) += eps;
                MAT_AT(g.bs[k], i, j) = (nn_cost(nn, ti, to) - c) / eps;
                MAT_AT(nn.bs[k], i, j) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t k = 0; k < nn.count; k++)
    {
        for (size_t i = 0; i < nn.ws[k].rows; i++)
        {
            for (size_t j = 0; j < nn.ws[k].cols; j++)
            {
                MAT_AT(nn.ws[k], i, j) -= rate * MAT_AT(g.ws[k], i, j);
            }
        }
        for (size_t i = 0; i < nn.bs[k].rows; i++)
        {
            for (size_t j = 0; j < nn.bs[k].cols; j++)
            {
                MAT_AT(nn.bs[k], i, j) -= rate * MAT_AT(g.bs[k], i, j);
            }
        }
    }
}

#endif // NN_IMPLEMENTATION