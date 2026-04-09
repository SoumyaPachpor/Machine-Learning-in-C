#include <stdio.h>
#include <stdlib.h>

float train_data[][2] ={
    {0.0, 0.0},
    {1.0, 2.0},
    {2.0, 4.0},
    {3.0, 6.0},
    {4.0, 8.0},
    {5.0, 10.0},
    {6.0, 12.0},
};
#define train_size sizeof(train_data)/sizeof(train_data[0])

float rand_float() {
    return (float) rand()/(float) RAND_MAX;
}
float cost(float w, float b){
    float result = 0.0;
    for(size_t i = 0; i < train_size; i++) {
        float x = train_data[i][0];
        float y = w*x + b;
        float d = y - train_data[i][1];
        result += d*d;
    }
    return result /= train_size;
}

int main(void) {
    srand(69);
    float w = rand_float()*10.0f;
    float b = rand_float()*5.0f;
    float eps = 1e-3;
    float r = 1e-2;
    for(size_t i = 0; i < 1000; i++){
        float c = cost(w, b);
        float dw = (cost(w+eps, b) - c)/eps;
        float db = (cost(w, b+eps) - c)/eps;
        w-=r*dw;
        b-=r*db;
        printf("cost: %f, w: %f, b: %f\n", c, w, b);
    }
    printf("weight: %f\n", w);
    return 0;
}