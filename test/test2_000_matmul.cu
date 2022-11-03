// CUDA runtime 库 + CUBLAS 库
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <stdlib.h>
#include <stdio.h>



int main()
{
    // 定义状态变量
    cublasStatus_t status;
    int batch_size = 2;
    int ch_in = 3;
    int ch_out = 4;
    float *h_X,*h_W,*h_Y;   //存储于内存中的矩阵
    h_X = (float*)malloc(sizeof(float)*batch_size*ch_in);  //在内存中开辟空间
    h_W = (float*)malloc(sizeof(float)*ch_in*ch_out);
    h_Y = (float*)malloc(sizeof(float)*batch_size*ch_out);

/*
X = [[0.11766736, 0.85902978, 0.5823179], [0.89798578, 0.36616095, 0.99145885]]
W = [[0.10423932, 0.85613949, 0.38102097, 0.98479858], [0.62318701, 0.60919121, 0.12577583, 0.14188402], [0.00613438, 0.80014461, 0.40659438, 0.24330399]]
Y_true = [[0.55117392, 1.0899916, 0.3896461, 0.37944151], [0.32787416, 1.78517358, 0.7913272, 1.17751341]]


计算Y = X x W
X.shape = (batch_size, ch_in)
W.shape = (ch_in, ch_out)
Y.shape = (batch_size, ch_out)


cublas中实际计算的是
Y^T = (X x W)^T = W^T x X^T = U x V
Y^T.shape = (ch_out, batch_size)
左矩阵是B^T，右矩阵是X^T。因为结果Y^T是列优先，所以实际得到的是Y。


cd test
nvcc -l cublas -L "C://Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64" test2_000_matmul.cu -o test2_000_matmul

./test2_000_matmul


*/
    int p = 0;
    h_X[p++] = 0.11766736;
    h_X[p++] = 0.85902978;
    h_X[p++] = 0.5823179;
    h_X[p++] = 0.89798578;
    h_X[p++] = 0.36616095;
    h_X[p++] = 0.99145885;
    p = 0;
    h_W[p++] = 0.10423932;
    h_W[p++] = 0.85613949;
    h_W[p++] = 0.38102097;
    h_W[p++] = 0.98479858;
    h_W[p++] = 0.62318701;
    h_W[p++] = 0.60919121;
    h_W[p++] = 0.12577583;
    h_W[p++] = 0.14188402;
    h_W[p++] = 0.00613438;
    h_W[p++] = 0.80014461;
    h_W[p++] = 0.40659438;
    h_W[p++] = 0.24330399;


    // 打印待测试的矩阵
    printf("X = ");
    for (int i=0; i<batch_size*ch_in; i++){
        printf("%f, ", h_X[i]);
    }
    printf("\n");
    printf("W = ");
    for (int i=0; i<ch_in*ch_out; i++){
        printf("%f, ", h_W[i]);
    }
    printf("\n");

    float *d_X,*d_W,*d_Y;    //存储于显存中的矩阵
    cudaMalloc((void**)&d_X,sizeof(float)*batch_size*ch_in); //在显存中开辟空间
    cudaMalloc((void**)&d_W,sizeof(float)*ch_in*ch_out);
    cudaMalloc((void**)&d_Y,sizeof(float)*batch_size*ch_out);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(d_X, h_X, sizeof(float)*batch_size*ch_in, cudaMemcpyHostToDevice); //数据从内存拷贝到显存
    cudaMemcpy(d_W, h_W, sizeof(float)*ch_in*ch_out, cudaMemcpyHostToDevice);

/*
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
计算C = alpha * op(A) * op(B) + beta * C
op(A).shape = (m, k)
op(B).shape = (k, n)
C.shape = (m, n)
m是指 经过op()作用的左矩阵的行数 或者 结果矩阵的行数；
n是指 经过op()作用的右矩阵的列数 或者 结果矩阵的列数；
k是指 经过op()作用的左矩阵的列数 或者 经过op()作用的右矩阵的行数；
如果左矩阵是列主序，lda填入未经过op()作用的左矩阵的列数，如果左矩阵是行主序，lda填入未经过op()作用的左矩阵的行数；
如果右矩阵是列主序，ldb填入未经过op()作用的右矩阵的列数，如果右矩阵是行主序，ldb填入未经过op()作用的右矩阵的行数；
结果矩阵只能是列主序，ldc填入结果矩阵的列数；
*/



/*
计算
Y = X x W = op(X) * op(W)
X^T.shape = (ch_in, batch_size)
W^T.shape = (ch_out, ch_in)
Y^T.shape = (ch_out, batch_size)

左矩阵op(X)是X，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，lda填入 X^T 的行数 ch_in
右矩阵op(W)是W，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，ldb填入 W^T 的行数 ch_out
结果矩阵是Y^T（可以理解为结果矩阵Y^T传出cublasSgemm()之前会强制转置1次得到Y）只能是列主序，ldc填入结果矩阵Y^T的列数 batch_size
*/
    float alpha = 1, beta = 0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                batch_size, ch_out, ch_in,
                &alpha,
                d_X, ch_in,
                d_W, ch_out,
                &beta,
                d_Y, batch_size);
    printf("C^T = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");

/*
计算
Y^T = (X x W)^T = W^T x X^T = op(W) * op(X)
W.shape = (ch_in, ch_out)
X.shape = (batch_size, ch_in)
Y.shape = (batch_size, ch_out)

左矩阵op(W)是W^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，lda填入 W 的列数 ch_out
右矩阵op(X)是X^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，ldb填入 X 的列数 ch_in
结果矩阵是Y（可以理解为结果矩阵Y传出cublasSgemm()之前会强制转置1次得到Y^T）只能是列主序，ldc填入结果矩阵Y的列数 ch_out
*/
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                ch_out, batch_size, ch_in,
                &alpha,
                d_W, ch_out,
                d_X, ch_in,
                &beta,
                d_Y, ch_out);
    printf("C   = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");


/*
实现pytorch的全连接层（W以转置的形式进行矩阵乘法）。
pytorch的全连接层权重W.shape = (ch_out, ch_in)，有点反人类。
这里先修改权重为权重的转置。
*/
    p = 0;
    h_W[p++] = 0.10423932;
    h_W[p++] = 0.62318701;
    h_W[p++] = 0.00613438;
    h_W[p++] = 0.85613949;
    h_W[p++] = 0.60919121;
    h_W[p++] = 0.80014461;
    h_W[p++] = 0.38102097;
    h_W[p++] = 0.12577583;
    h_W[p++] = 0.40659438;
    h_W[p++] = 0.98479858;
    h_W[p++] = 0.14188402;
    h_W[p++] = 0.24330399;
    cudaMemcpy(d_W, h_W, sizeof(float)*ch_in*ch_out, cudaMemcpyHostToDevice);

/*
计算 （注意，其中的W是pytorch全连接层风格的W）
Y = X x W^T = op(X) * op(W)
X^T.shape = (ch_in, batch_size)
W.shape = (ch_out, ch_in)
Y^T.shape = (ch_out, batch_size)

左矩阵op(X)是X，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，lda填入 X^T 的行数 ch_in
右矩阵op(W)是W^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，ldb填入 W 的列数 ch_in
结果矩阵是Y^T（可以理解为结果矩阵Y^T传出cublasSgemm()之前会强制转置1次得到Y）只能是列主序，ldc填入结果矩阵Y^T的列数 batch_size
*/
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                batch_size, ch_out, ch_in,
                &alpha,
                d_X, ch_in,
                d_W, ch_in,
                &beta,
                d_Y, batch_size);
    printf("C^T = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");


/*
计算 （注意，其中的W是pytorch全连接层风格的W）
Y^T = (X x W^T)^T = W x X^T = op(W) * op(X)
W^T.shape = (ch_in, ch_out)
X.shape = (batch_size, ch_in)
Y.shape = (batch_size, ch_out)

左矩阵op(W)是W，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，lda填入 W^T 的行数 ch_in
右矩阵op(X)是X^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，ldb填入 X 的列数 ch_in
结果矩阵是Y（可以理解为结果矩阵Y传出cublasSgemm()之前会强制转置1次得到Y^T）只能是列主序，ldc填入结果矩阵Y的列数 ch_out
*/
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                ch_out, batch_size, ch_in,
                &alpha,
                d_W, ch_in,
                d_X, ch_in,
                &beta,
                d_Y, ch_out);
    printf("C   = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");



/*
实现X以转置的形式进行矩阵乘法。
pytorch的全连接层权重W.shape = (ch_out, ch_in)，有点反人类。
这里先修改权重为权重的转置。
*/
    p = 0;
    h_X[p++] = 0.11766736;
    h_X[p++] = 0.89798578;
    h_X[p++] = 0.85902978;
    h_X[p++] = 0.36616095;
    h_X[p++] = 0.5823179;
    h_X[p++] = 0.99145885;
    p = 0;
    h_W[p++] = 0.10423932;
    h_W[p++] = 0.85613949;
    h_W[p++] = 0.38102097;
    h_W[p++] = 0.98479858;
    h_W[p++] = 0.62318701;
    h_W[p++] = 0.60919121;
    h_W[p++] = 0.12577583;
    h_W[p++] = 0.14188402;
    h_W[p++] = 0.00613438;
    h_W[p++] = 0.80014461;
    h_W[p++] = 0.40659438;
    h_W[p++] = 0.24330399;
    cudaMemcpy(d_X, h_X, sizeof(float)*batch_size*ch_in, cudaMemcpyHostToDevice); //数据从内存拷贝到显存
    cudaMemcpy(d_W, h_W, sizeof(float)*ch_in*ch_out, cudaMemcpyHostToDevice);

/*
计算
Y = X^T x W = op(X) * op(W)
X.shape = (ch_in, batch_size)
W^T.shape = (ch_out, ch_in)
Y^T.shape = (ch_out, batch_size)

左矩阵op(X)是X^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，lda填入 X 的列数 batch_size
右矩阵op(W)是W，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，ldb填入 W^T 的行数 ch_out
结果矩阵是Y^T（可以理解为结果矩阵Y^T传出cublasSgemm()之前会强制转置1次得到Y）只能是列主序，ldc填入结果矩阵Y^T的列数 batch_size
*/
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size, ch_out, ch_in,
                &alpha,
                d_X, batch_size,
                d_W, ch_out,
                &beta,
                d_Y, batch_size);
    printf("C^T = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");

/*
计算
Y^T = (X^T x W)^T = W^T x X = op(W) * op(X)
W.shape = (ch_in, ch_out)
X^T.shape = (batch_size, ch_in)
Y.shape = (batch_size, ch_out)

左矩阵op(W)是W^T，不需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以不需要转置），是列主序，lda填入 W 的列数 ch_out
右矩阵op(X)是X，需要转置（可以理解为矩阵传入cublasSgemm()后会强制转置1次，所以需要转置），是行主序，ldb填入 X^T 的行数 batch_size
结果矩阵是Y（可以理解为结果矩阵Y传出cublasSgemm()之前会强制转置1次得到Y^T）只能是列主序，ldc填入结果矩阵Y的列数 ch_out
*/
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                ch_out, batch_size, ch_in,
                &alpha,
                d_W, ch_out,
                d_X, batch_size,
                &beta,
                d_Y, ch_out);
    printf("C   = ");
    cudaMemcpy(h_Y, d_Y, sizeof(float)*batch_size*ch_out, cudaMemcpyDeviceToHost);
    for(int i=0;i<batch_size*ch_out;++i) {
        printf("%f, ", h_Y[i]);
    }
    printf("\n");




    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    free(h_X);
    free(h_W);
    free(h_Y);
    return 0;
}
