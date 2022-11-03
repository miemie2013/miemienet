#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <vector>
//#include <chrono>
//#include <immintrin.h>


void init_ptr(float* ptr, int len, float value)
//void init_ptr(float* ptr, int len, float value, int num_threads_)
{
//    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < len; i++)
    {
        *(ptr + i) = value;
    }
}

int main(int argc, char** argv)
{
/*
g++ test/test2_00000_gcc.cpp -fopenmp -march=native -o test2_00000_gcc_fast.out -w -Ofast

g++ test/test2_00000_gcc.cpp -fopenmp -march=native -o test2_00000_gcc.out -w

./test2_00000_gcc_fast.out

./test2_00000_gcc.out

objdump -d ./test2_00000_gcc_fast.out > test2_00000_gcc_fast.txt

objdump -d ./test2_00000_gcc.out > test2_00000_gcc.txt

g++ -S test/test2_00000_gcc.cpp -fopenmp -march=native -o test2_00000_gcc_fast.out -w -Ofast



g++ -S test/test2_00000_gcc.cpp -o test2_00000_gcc.s -fverbose-asm

g++ -S test/test2_00000_gcc.cpp -o test2_00000_gcc_fast.s -Ofast -fverbose-asm


g++ -S test/test2_00000_gcc.cpp -fopenmp -march=native -o test2_00000_gcc_omp.s -fverbose-asm

g++ -S test/test2_00000_gcc.cpp -fopenmp -march=native -o test2_00000_gcc_omp_fast.s -Ofast -fverbose-asm



*/
    int num_threads_ = 12;
    int M = 13;
    int bytes = sizeof(float) * M;
    float* out_true = (float*) malloc(bytes);
    float* out = (float*) malloc(bytes);

//    init_ptr(out, M, 3.3f, num_threads_);
    init_ptr(out, M, 3.3f);


    return 0;
}