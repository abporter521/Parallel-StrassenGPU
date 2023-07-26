#include "matrix.h"
#include <cuda.h>

__global__ void MatrixAdd(Matrix A, Matrix B, Matrix&C);

// Adds two matrices together. If there is an error in dimension matching
void callMatrixAdd(int size, Matrix A, Matrix B, Matrix&C){
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(size/ threadsPerBlock.x, size/ threadsPerBlock.y);
    MatrixAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);
}
__global__ void MatrixAdd(Matrix A, Matrix B, Matrix&C){
    if(A.height != B.height || A.width != B.width){
        return;
    }
    
    int x = threadIdx.x;
    int y = threadIdx.y;

    if(x < A.width && y < A.height){
        C.values[x*A.width + y] = A.values[x*A.width + y] + B.values[x*A.width + y];
    }

   return;
}