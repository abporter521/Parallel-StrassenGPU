#include "matrix.h"
#include <cuda.h>

__global__ void MatrixAdd(Matrix A, Matrix B, Matrix&C);


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