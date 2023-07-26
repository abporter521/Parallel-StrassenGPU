/*
* This is the final project for CSCE 735 Parallel Computing
* The goal of this project is to write a matrix multiplier program
* that will multiply 2 matrices of dimension n x n, where n = 2^k.
* The program will utilize CUDA programming to speed up execution of 
* Strassen's Algorithm for matrix multiplication

* Author: Andrew Porter, 933007811
* Date: 20 July 2023
*/
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <new>
#include "matrix.h"

#define DEBUG 1



// Function prototype for StrassensAlgorithm
//Matrix StrassensAlgorithm(const Matrix& A, const Matrix& B, int size);
__global__ void MatrixAdd(Matrix A, Matrix B, Matrix& C);

/*
__global__ void MatrixSubtract(Matrix A, Matrix B, Matrix C);
__global__ void initializeMatrix(Matrix& mat, int height, int width);



__global__ void MatrixSubtract(Matrix A, Matrix B, Matrix C){
    if(A.height != B.height || A.width != B.width){
        printf("Runtime Error: Matrix Dimensions do not match for subtraction");
        exit(1);
    }
    
    int x = threadID.x;
    int y = threadID.y;

    if(x < A.width && y < A.height){
        C.values[x*A.width + y] = A.values[x*A.width + y] - B.values[x*A.width + y];
    }

   return;
}

// Function splits matrix into 4 submatrices
/* ________
   |a1| a2|
   --------
   |b1| b2|
   --------
*//*
void MatrixSplit(const Matrix A, Matrix* a1, Matrix* a2, Matrix* b1, Matrix* b2){
    int size = A.height/2;

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            a1->values[i*A.width + j] = A.values[i*A.width + j];
            a2->values[i*A.width + j] = A.values[i*A.width + j + size];
            b1->values[i*A.width + j] = A.values[(i+size)*A.width + j];
            b2->values[i*A.width + j] = A.values[(i+size)*A.width + j + size];
        }
    }
    return;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    
    C.elements[row * C.width + col] = Cvalue;
}

// Compute the matrix product by brute force while 
// leveraging the benefits of threads on a system.
Matrix MatrixMultiply(Matrix A, Matrix B){
    if(A.height != B.width){
        printf("Runtime Error: Matrix Dimensions do not match for multiplication");
        exit(1);
    }
    
    // Thread needs a row and a column
    // Load A and B to device memory
      Matrix dA;
    dA.height = A.height; dA.width=A.width;
    size_t sizeA = dA.height * dA.width * sizeof(float);
    cudaMalloc(&dA.values, sizeA);
    cudaMemcpy(dA.values, A.values, sizeA, cudaMemcpyHostToDevice);
    
    Matrix dB;
    dB.height = B.height; dB.width=B.width;
    size_t sizeB = dB.height * dB.width * sizeof(float);
    cudaMalloc(&dA.values, sizeA);
    cudaMemcpy(dB.values, B.values, sizeB, cudaMemcpyHostToDevice);

    Matrix dC;
    dC.height = B.height; dC.width=A.width;
    size_t sizeC = dC.height * dC.width * sizeof(float);
    cudaMalloc(&dC.values, sizeC);

   
   // Set up Blocks
   // Call the multiply kernel

   // Output C

}
// This function is designed to start Strassen's Algorithm
// Returns product
Matrix StrassensAlgorithm(const Matrix&A, const Matrix&B, int size){

    /*
    if size is <= min size
        Return Brute Force of A and B using the benefit of threads in a block
    *//*
  
    //Split group into block
    int rsize = size/2;
    Matrix A11;
    Matrix B11;
    Matrix A12, B12;
    Matrix A21, B21;
    Matrix A22, B22;
    initializeMatrix(A11, rsize, rsize);
    initializeMatrix(B11, rsize, rsize);
    initializeMatrix(A11, rsize, rsize);
    initializeMatrix(B11, rsize, rsize);
    initializeMatrix(A11, rsize, rsize);
    initializeMatrix(B11, rsize, rsize);
    initializeMatrix(A11, rsize, rsize);
    initializeMatrix(B11, rsize, rsize);
    SplitMat(A, &A11, &A12, &A21, &A22);
    SplitMat(B, &B11, &B12, &B21, &B22);

    
    //What if instead of trying to recursively make smaller, for right now I just need to create the submatrices with blocks. So 
    // If smallest Size is 2 x 2 matrix or something, I will just create enough blocks so that when divided up, we get 2 x 2 matrix
    //in each block. Then it should be easier to combine with neighbor blocks to build larger blocks

    // Compute the different parts
    Matrix M1 = StrassensAlgorithm(MatrixAdd(A11, A22), MatrixAdd(B11,B22), size/2);
    Matrix M2 = StrassensAlgorithm(MatrixAdd(A21,A22), B11, size/2);
    Matrix M3 = StrassensAlgorithm(A11, MatrixSubtract(B12,B22), size/2);
    Matrix M4 = StrassensAlgorithm(A22, MatrixSubtract(B21,B11), size/2);
    Matrix M5 = StrassensAlgorithm(MatrixAdd(A11,A12), B22,size/2);
    Matrix M6 = StrassensAlgorithm(MatrixSubtract(A21,A11), MatrixAdd(B11,B12), size/2);
    Matrix M7 = StrassensAlgorithm(MatrixSubstract(A12,A22),(B21,B22), size/2);

    // Turn into Block Splits
    Matrix C11 = MatrixSubtract(MatrixAdd(M1,M4),MatrixAdd(M5,M7));
    Matrix C12 = MatrixAdd(M3,M5);
    Matrix C21 = MatrixAdd(M2,M4);
    Matrix C22 = MatrixSubtract(M1, MatrixAdd(MatrixAdd(M2,M3),M6));

    CombineMatrix(&dC, C11, C12,C21,C22);
}
*/
// Function to initialize a Matrix
void initializeMatrix(Matrix& mat, int height, int width, int fill) {
    mat.height = height;
    mat.width = width;

    // Allocate memory for the values array
    mat.values = new float[height * width];

    // For example, if you want to initialize all elements to 0:
    if(fill){
        for (int i = 0; i < height * width; ++i) {
            mat.values[i] = rand();
        }
    }
    else{
        for (int i = 0; i < height * width; ++i) {
            mat.values[i] = 0.0f;
        }
    }
    return;
}

// Function to free the memory of a Matrix
void freeMatrix(Matrix& mat) {
    delete[] mat.values;
}

void printMatrix(Matrix m){
    for (int i = 0; i < m.height; ++i) {
        for (int j = 0; j < m.width; ++j) {
            printf("%f  ", m.values[i * m.width + j]);
        }
        printf("\n");
    }
    return;
}
// Main entry into the program
int main(int argc, char* argv[]){

// User inputs
int size;
int min_size;

if(argc != 3){
    printf("Argument must be 2 integers <k>, <k'> where each is a power of 2");
}

size = std::atoi(argv[1]);
min_size = std::atoi(argv[2]);

srand(time(0));
// TODO: Check that sizes are powers of 2

// TODO: Initialize and allocate memory for A, B, C
Matrix A, B, C;
initializeMatrix(A, size, size,1);
initializeMatrix(B, size, size,0);
initializeMatrix(C, size, size, 0);

if(DEBUG){
    printf("This is A:\n");
    printMatrix(A);
    printf("This is B:\n");
    printMatrix(B);




    // Begin Test ADD and Subtract
    printf("Beginning Addition Tests");
   // dim3 threadsPerBlock(16,16);
    //dim3 numBlocks(size/ threadsPerBlock.x, size/ threadsPerBlock.y);
    //MatrixAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);

    //cudaDeviceSynchronize(); 
    
    printf("This is Matrix C:\n");
    printMatrix(C);
}

// START TIMER
// TODO: Begin Strassen's Algorithm
//

// END TIMER

//CleanUP
freeMatrix(A);
freeMatrix(B);

}
