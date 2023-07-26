#ifndef MATRIX_H
#define MATRIX_H

struct Matrix{
    int height;
    int width;
    float* values;
};

#endif

__global__ void MatrixAdd(Matrix A, Matrix B, Matrix&C);