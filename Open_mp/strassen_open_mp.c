// OpenMP implementation of Strassen's Algorithm for CSCE 735 Major Project
// 
// TODO: Verify that changing width to mat_size in all helper functions did not wreck my program
// TODO: Implement Strassen Algorithm
//
// AUTHOR: Andrew Porter
// VERSION: 2.0 Rearranged code to implement Matrix Multiplication
// VERSION: 1.0 Implemented OpenMP multi-threading commands to run in parallel. 
//
/* COMMON BUGS: 
Error in `./strassen.exe': corrupted size vs. prev_size: 0x000000000108b330 ***
FIX: Had to modify Add and subtract functions because they were writing out of bounds
    C matrix was allocated size x size memory, but was writing outside bounds because size < mat_size
    mat_size was the original offset I was writing to.

    Error in Output of C11 and C22; C12, and C22 are correct 
    FIX: Order of operations does matter. Apparently (P1 + P4) - (P5 + P7) != P1 + P4 - P5 + P7

*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>


#define MAX_THREADS     65536
#define MAX_LIST_SIZE   INT_MAX

#define PRINT_MATRIX 0

#define DEBUG 0
#define DEBUG_S 0
#define TEST_ADD 0
#define TEST_SUB 0
#define TEST_VERIFY 0
#define TEST_BRUTE 0
#define TEST_SPLIT 0
#define TEST_SPLIT_OPS 0
#define TEST_STRASSEN_THRESH 0
#define TEST_SPLIT_WHOLE_OPS 1

// Global variables
int thresh_size;		 
int mat_size;
int num_threads;		
int *A;			
int *B;			
int *C;
int *D;	


// Functions
int* Strassen_Multiply(int* A, int* B, int width);
int* Add_Matrix(const int* A, const int* B, int width);
int* Sub_Matrix(const int* A, const int* B, int width);
int* Mult_Brute(const int* A, const int* B, int width, int newMat);
void Split_Matrix(int* A, int** A11,  int** A12, int** A21, int** A22, int size);
int* Copy_Matrix(int* subA, int width);
int* Place_Submatrix(int* C11, int* C12, int* C21, int* C22, int width);
int* initializeMatrix(int size, int fill);
void printMatrix(const int* m, int size, int newMat);
void clearMatrix(int* m, int size);
int Mult_Verify(const int* A, const int* B, int width);

// Main entry into the program
int main(int argc, char* argv[]){
    // User inputs
    if(argc != 4){
        printf("Argument must be 2 integers <k>, <k'> where each is a power of 2\n 3rd argument is number of threads to use");
        return 1;
    }

    struct timespec start, stop, stop_brute;
    double total_time, time_res, total_time_brute;

    //Get matrix size and threshold size from user
    mat_size = 1 << (atoi(argv[1]));
    thresh_size = 1 << (atoi(argv[2]));
    num_threads = atoi(argv[3]);
    srand(time(0));
    
    // Initialize and allocate memory for A, B, C, D
    A = initializeMatrix(mat_size,1);
    B = initializeMatrix(mat_size,1);
      
    if(PRINT_MATRIX){
        printf("This is A:\n");
        printMatrix(A, mat_size, 0);
        printf("This is B:\n");
        printMatrix(B, mat_size, 0);
    }
    if(DEBUG){
        D = initializeMatrix(mat_size, 0);
        int* smaller_guy;
        int* top_left, *top_right, *bottom_left, *bottom_right;

        if(TEST_ADD){
            printf("Testing Addition:\n");
            C = Add_Matrix(A, B, mat_size);
            printf("This is Sum of A and B:\n");
            printMatrix(C, mat_size, 0);
            clearMatrix(C, mat_size);
            printf("\n");
            free(C);
        }
        if(TEST_SUB){
            printf("Testing Subtraction:\n");
            C = Sub_Matrix(A, B, mat_size);
            printf("This is Difference of A and B:\n");
            printMatrix(C, mat_size, 0);
            free(C);
            printf("\n");
        }
        if(TEST_VERIFY){
            int flag = 0;
            printf("\nTesting Verify if A != B\n");
            
            if(Mult_Verify(A, B, mat_size)==0){
                printf("Verification Failed: A & B are not equal\n"); flag = 1;
            }

            printf("Testing Verify Copy A to C:\n");
            C = Add_Matrix(A, D, mat_size);
            printf("Matrix A:\n");
            printMatrix(A, mat_size, 0);
            printf("Matrix C-Copied from A:\n");
            printMatrix(C, mat_size, 0);

            printf("Testing Verify if A = C\n");
            if(Mult_Verify(A, C, mat_size)){
                printf("Verification Failed: A & C are equal, but reported not equal\n"); flag = 1;
            }
           
            if(flag==0)
                printf("Verification passed\n\n");
            free(C);
        }
        if(TEST_BRUTE){
            printf("Testing Brute Force multiplication:\n");
            C = Mult_Brute(A, B, mat_size, 1);
            printf("This is the product of A and B:\n");
            printMatrix(C, mat_size, 0);
            printf("\n");
            free(C);
        }
        if(TEST_SPLIT){  
            printf("Splitting Matrix A\n");      
            Split_Matrix(A, &top_left, &top_right, &bottom_left, &bottom_right, mat_size);
            printf("This is submatrix A12:\n");
            printMatrix(top_right, mat_size/2, 0);
            printf("\n");
        }
        if(TEST_SPLIT_OPS){
            // Test Adding After splitting
            printf("Testing Addition and Multiplication of submatrices:\n");
            printMatrix(A, mat_size, 0);
            
            // Split Matrix A into 4 parts
            Split_Matrix(A, &top_left, &top_right, &bottom_left, &bottom_right, mat_size);
            int * tl = Copy_Matrix(top_left, mat_size);
            int * tr = Copy_Matrix(top_right, mat_size);
            printf("Copying...\n");
            printf("Top left\n");
            printMatrix(tl, mat_size/2, 1);
            printf("Top right\n");
            printMatrix(tr, mat_size/2, 1);

            // Add the two smaller submatrices together
            smaller_guy = Add_Matrix(tl,tr, mat_size/2);
            printf("This is the sum of A11 and A12:\n");
            printMatrix(smaller_guy, mat_size/2, 1);
            free(smaller_guy);
            
            free(tr);free(tl);
            printf("This is the brute product of A21 and A22:\n");
            printMatrix(bottom_left, mat_size/2, 0);
            printf("*\n");
            printMatrix(bottom_right,mat_size/2, 0);
            printf("=\n");
            smaller_guy = Mult_Brute(bottom_left, bottom_right, mat_size/2, 0);
            printMatrix(smaller_guy, mat_size/2, 1);
            printf("\n");
            free(smaller_guy);
            printf("Finished Split_Ops Test\n"); 
        }
        if(TEST_STRASSEN_THRESH){
            if(mat_size != thresh_size)
                printf("Skipping Strassen Thresh Test. Sizes are not equal\n\n");
            else{
                printf("Starting Strassen Threshold Test:\n");
                C = Strassen_Multiply(A, B, mat_size);
                printMatrix(C, mat_size, 0);
                int * p = Mult_Brute(A, B, mat_size, 0);
                if(Mult_Verify(C, p, mat_size)){
                    printf("Pointer Returned from Strassen Failed\n");
                }
                else{
                    printf("Threshold Test passed!\n");
                }
                free(C);
                free(p);
            }
            
            clearMatrix(D, mat_size);
        }
        if(TEST_SPLIT_WHOLE_OPS){
            Split_Matrix(A, &top_left, &top_right, &bottom_left, &bottom_right, mat_size);
            int* Buddy = initializeMatrix(mat_size/2, 1);
            int* copy = Copy_Matrix(top_right, mat_size);
            int* product = Mult_Brute(copy, Buddy, mat_size/2, 1);
            printMatrix(copy, mat_size/2, 1);
            printf("\n");
            printMatrix(Buddy, mat_size/2, 1);
            printf("\n");
            printMatrix(product, mat_size/2, 1);
            free(Buddy); free(copy); free(product);
        }
        free(D);
        printf("\nEnd of Tests\n");
    }
    
    clock_gettime(CLOCK_REALTIME, &start);
    // Call Strassen
    C = Strassen_Multiply(A, B, mat_size);

    // Compute time taken
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)	+0.000000001*(stop.tv_nsec-start.tv_nsec);

    int * brute = Mult_Brute(A, B, mat_size, 0);
    clock_gettime(CLOCK_REALTIME, &stop_brute);
    total_time_brute = (stop_brute.tv_sec-stop.tv_sec)+0.000000001*(stop_brute.tv_nsec-stop.tv_nsec); 

    if(DEBUG){
        if(C==NULL)
            printf("FAIL: Returned from Strassen But answer is NULL\n");
        else{
            if(Mult_Verify(C, brute, mat_size) == 0){
                if(PRINT_MATRIX){
                    printf("Fantastic!!!\nThis is the returned Matrix C:\n");
                    printMatrix(C, mat_size, 1);
                }
            }
            else
                printf("FAIL: Matrix C is incorrect\n");       
    }
    }

    // Print time taken
    printf("Matrix Size = %d^2, Threads = %d, time (sec) = %8.4f, brute_time = %8.4f\n", 
	    mat_size, num_threads, total_time, total_time_brute);
    free(C);
    free(brute);
 
    // Cleanup
    free(A); free(B); 
}

// Strassen Algorithm Main
int* Strassen_Multiply(int* A, int* B, int width){ 
    int *ans;
    
    if(width <= thresh_size){ 
        ans = Mult_Brute(A, B, width, 1); 
        return  ans;
    }

    int newsize = width/2;
    int *C11, *C12, *C21, *C22;
    int* P1, *P2, *P3, *P4, *P5, *P6, *P7;
    // Declare Submatrices
    /*A11 = A_subs[0] A12 = A_subs[1] A21 = A_subs[2] A22 = A_subs[3]*/
    int** A_subs = malloc(4*sizeof(int*));
    int** B_subs = malloc(4*sizeof(int*));
    
    // Split The Matrix and copy over
    Split_Matrix(B, &B_subs[0], &B_subs[1], &B_subs[2], &B_subs[3], width);
    Split_Matrix(A, &A_subs[0], &A_subs[1], &A_subs[2], &A_subs[3], width);
    int* A11= Copy_Matrix(A_subs[0], width);int* A12= Copy_Matrix(A_subs[1], width);int* A21= Copy_Matrix(A_subs[2], width);int* A22= Copy_Matrix(A_subs[3], width);
    int* B11= Copy_Matrix(B_subs[0], width);int* B12= Copy_Matrix(B_subs[1], width);int* B21= Copy_Matrix(B_subs[2], width);int* B22= Copy_Matrix(B_subs[3], width);

    free(A_subs);
    free(B_subs);

    if(DEBUG_S){          
        printf("New A_subs\n");
        printMatrix(A_subs[0], newsize, 0);
        printf("\n");
        printMatrix(A_subs[1], newsize, 0);
        printf("\n");
        printMatrix(A_subs[2], newsize, 0);printf("\n");
        printMatrix(A_subs[3], newsize, 0);printf("\n\n");
        printf("New B_subs\n");
        printMatrix(B_subs[0], newsize, 0);
        printf("\n");
        printMatrix(B_subs[1], newsize, 0);
        printf("\n");
        printMatrix(B_subs[2], newsize, 0);printf("\n");
        printMatrix(B_subs[3], newsize, 0);printf("\n\n");
    }
    
    omp_set_max_active_levels(1);
    #pragma omp parallel default(shared) num_threads(num_threads)
    {
        #pragma omp sections
        {
            //Calculate Subparts
            #pragma omp section
            {
                int* P11 = Add_Matrix(A11, A22, newsize);
                int* P12 = Add_Matrix(B11, B22, newsize);
                P1 = Strassen_Multiply(P11, P12, newsize);
                free(P11); free(P12);
            }
            #pragma omp section
            {
                int* P21 = Add_Matrix(A21, A22, newsize);
                P2 = Strassen_Multiply(P21, B11, newsize);
                free(P21);
            }
            #pragma omp section
            {
                int* P31 = Sub_Matrix(B12, B22, newsize); 
                P3 = Strassen_Multiply(A11, P31, newsize);
                free(P31);
            }
            #pragma omp section
            {
                int* P41 = Sub_Matrix(B21, B11, newsize);
                P4 = Strassen_Multiply(A22, P41, newsize);
                free(P41);
            }
            #pragma omp section
            {
                int* P51 = Add_Matrix(A11, A12, newsize);
                P5 = Strassen_Multiply(P51, B22, newsize);
                free(P51);
            }
            #pragma omp section
            {
                int* P61 = Sub_Matrix(A21, A11, newsize); 
                int* P62 = Add_Matrix(B11, B12, newsize);
                P6 = Strassen_Multiply(P61, P62, newsize);
                free(P61); free(P62);
            }
            #pragma omp section
            {
                int* P71 = Sub_Matrix(A12, A22, newsize); 
                int* P72 = Add_Matrix(B21, B22, newsize);
                P7 = Strassen_Multiply(P71, P72, newsize);
                free(P71); free(P72);  
            } 
        }
              
        #pragma omp sections
        {
            #pragma omp section
            {
            //Order of Operations Matters!!!
            int* C111 = Add_Matrix(P1, P4, newsize);
            int* C112 = Sub_Matrix(C111, P5, newsize);
            C11 = Add_Matrix(C112, P7, newsize);
            free(C111); free(C112);
            }
            #pragma omp section
            {
            C12 = Add_Matrix(P3, P5, newsize); 
            }
            #pragma omp section
            {
            C21 = Add_Matrix(P2, P4, newsize);
            }

            #pragma omp section
            {
            int* C221 = Sub_Matrix(P1, P2, newsize);
            int* C222 = Add_Matrix(C221, P3, newsize);
            C22 = Add_Matrix(C222, P6, newsize);
            free(C221); free(C222);
            }
        }
    }
    free(P1);free(P2);
    free(P3);free(P4);
    free(P5);free(P6);
    free(P7);       

    if(DEBUG_S){ 
        printf("These are my C matrices:\n");
        printMatrix(C11, newsize, 1);
        printf("\n");
        printMatrix(C12, newsize, 1);
        printf("\n");
        printMatrix(C21, newsize, 1);
        printf("\n");
        printMatrix(C22, newsize, 1);
        printf("\n");
    }
    
    ans = Place_Submatrix(C11, C12, C21, C22, width);
    // Cleanup
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(C11); free(C21); free(C12); free(C22);  

    return ans;
}

// Adds two square matrices of dimension width * width
// Bug is that I am adding to a new c matrix that is of size width != mat_size. Assumes that I am
//Adding 2 matrices of size mat_size
int* Add_Matrix(const int* A, const int* B, int width){
    
    int * C = initializeMatrix(width, 0);
    //printf("Initializing C matrix in Add\n");
    //printMatrix(A, width, 1);
    //printMatrix(B, width, 1);

    for(int i = 0; i < width; i++)
        for(int j = 0; j < width; j++){
            C[i * width + j] = A[i * width + j] + B[i * width + j];
        }
    
    return C;
}

// Subtracts two square matrices of dimension width * width
int* Sub_Matrix(const int* A, const int* B, int width){
    //Do a collapsed for loop here
    int * C = initializeMatrix(width, 0);

    for(int i = 0; i < width; i++)
        for(int j = 0; j < width; j++){
            C[i * width + j] = A[i * width + j] - B[i * width + j];
        }
    return C;
}

// Brute force multiplication of two matrices
int* Mult_Brute(const int* A, const int* B, int width, int newMat){
    int * C = initializeMatrix(width, 0);
    int offset;
    if (newMat)
        offset = width;
    else
        offset = mat_size;

    for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
            for(int k = 0; k < width; k++){
                C[i * width + j] += A[i * offset + k] * B[k * offset + j];
            }
        }
    }

    return C;
}

// Splits Matrix into 4 parts
// | A11 A12 |
// | A21 A22 |
void Split_Matrix(int* A, int** A11,  int** A12, int** A21, int** A22, int size) {
    // Calculate the size of each submatrix (N/2)
    int subSize = size / 2;

    *A11 = A;
    *A12 = A + subSize;
    *A21 = A + size * subSize;
    *A22 = A + size * subSize + subSize;
}

// Copy a submatrix to a new matrix in order to make it easier to multiply
// Width is size of A, which subA is a part of. SubA points to first index of submatrix
int* Copy_Matrix(int* subA, int width){
    int half = width/2;
    int * b = initializeMatrix(half, 0);

    for(int i = 0; i < half; i ++)
        for (int j = 0; j < half; j++)
            b[i * half + j] = subA[i * width + j];
    
    return b;
}

// Uses this to place submatrices into matrix
int* Place_Submatrix(int* C11, int* C12, int* C21, int* C22, int width){
    int * ans = initializeMatrix(width, 0);
    int half = width/2;
    
    #pragma omp parallel 
    {
        // Copy Over C11
        #pragma omp for collapse(2)
        for(int i = 0; i < half; i++){
            for(int j = 0; j < half; j++){
                ans[i*width + j] = C11[i * half + j];
            }
        }

        // Copy Over C12
        #pragma omp for collapse(2)
        for(int i = 0; i < half; i++){
            for(int j = 0; j < half; j++){
                ans[i*width + j + half] = C12[i * half + j];
            }
        }

        // Copy Over C21
        #pragma omp for collapse(2)
        for(int i = 0; i < half; i++){
            for(int j = 0; j < half; j++){
                ans[(i+half)*width + j] = C21[i * half + j];
            }
        }

        // Copy Over C22
        #pragma omp for collapse(2)
        for(int i = 0; i < half; i++){
            for(int j = 0; j < half; j++){
                ans[(i+half)*width + j + half] = C22[i * half + j];
            }
        }
    }
    return ans;
}

// Function to initialize a Matrix
int* initializeMatrix(int size, int fill) {
    int RANGE = 10;

    // Allocate memory for the values array
    int *mat = (int*) malloc(size*size*sizeof(int));

    // Fills matrix with 0s
    if(fill == 0){
        for (int i = 0; i < size*size; i++) {
            mat[i] = 0;
        }       
    }
    // Fills matrix with Randoms
    else{
        for (int i = 0; i < size*size; ++i) {
            mat[i] = rand() % RANGE;
        }
    }
    return mat;
}

// Prints out the Matrix
void printMatrix(const int* m, int size, int newMat){
    int offset;
    if (newMat)
        offset = size;
    else
        offset = mat_size;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int value = m[i * offset + j];
            if(value < 10 && value >= 0) // Adjusts columns for single digit
                printf("  %d  ", value);
            else if(value < 100)
                printf(" %d  ", value);
            else
                printf("%d  ", value);
        }
        printf("\n");
    }
    return;
}

// Fills matrix with 0s
void clearMatrix(int* m, int size){
    for(int i = 0; i < size; i ++){
        for(int j = 0; j < size; j ++)
            m[i*mat_size + j] = 0;
    }
}

// Compares two matrices. If they are equal, returns 0 otherwise 1.
int Mult_Verify(const int* A, const int* B, int width){
    for(int i = 0; i < width; i++)
        for(int j = 0; j < width; j++){
            if(A[i * width + j] != B[i * width + j])
                return 1;
        }
    return 0;
}
