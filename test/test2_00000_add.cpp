#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <vector>
//#include <chrono>
//#include <immintrin.h>


float add(float a, float b)
{
    return a+b;
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



g++ -S test/test2_00000_add.cpp -o test2_00000_add.s -fverbose-asm

g++ -S test/test2_00000_add.cpp -o test2_00000_add_fast.s -Ofast -fverbose-asm




*/
    float a = 1.3f;
    float b = 1.3f;
    float c = add(a, b);
    float d = add(2.5f, 2.6f);
    printf("c=%f\n", c);
    printf("d=%f\n", d);


    return 0;
}