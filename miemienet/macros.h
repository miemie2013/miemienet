#ifndef __MACROS_H__
#define __MACROS_H__

#if defined(WINDOWS)
    #define MM_DLL     __declspec(dllexport)
#else  // LINUX
    #define MM_DLL
#endif

#define NS_MM_BEGIN                     namespace miemienet {
#define NS_MM_END                       }
#define NS_MM_F_BEGIN                   namespace miemienet { namespace functional {
#define NS_MM_F_END                     }}
#define SNT                             (std::nothrow)

// Config data_format
#define NCHW 0
#define NHWC 1


// Tensor data type
#define INT64 0
#define INT32 1
#define INT8 2
#define FP32 3

// Layer forward type
#define SISO 0
#define SIMO 1
#define MISO 2
#define MIMO 3

// elementwise op
#define ELE_ADD 0
#define ELE_SUB 1
#define ELE_MUL 2
#define ELE_DIV 3
#define ELE_MIN 4
#define ELE_MAX 5

// reduce op
#define RED_SUM 0
#define RED_SUMSQUARE 2
#define RED_MEAN 3
#define RED_MAX 4
#define RED_MIN 5
#define RED_PROD 6
#define RED_L1 7
#define RED_L2 8
#define RED_LOGSUM 9
#define RED_LOGSUMEXP 10

// transpose op
#define TRANS2D_01 0
#define TRANS2D_10 1
#define TRANS3D_012 2
#define TRANS3D_021 3
#define TRANS3D_102 4
#define TRANS3D_120 5
#define TRANS3D_201 6
#define TRANS3D_210 7
#define TRANS4D_0123 8
#define TRANS4D_0132 9
#define TRANS4D_0213 10
#define TRANS4D_0231 11
#define TRANS4D_0312 12
#define TRANS4D_0321 13
#define TRANS4D_1023 14
#define TRANS4D_1032 15
#define TRANS4D_1203 16
#define TRANS4D_1230 17
#define TRANS4D_1302 18
#define TRANS4D_1320 19
#define TRANS4D_2013 20
#define TRANS4D_2031 21
#define TRANS4D_2103 22
#define TRANS4D_2130 23
#define TRANS4D_2301 24
#define TRANS4D_2310 25
#define TRANS4D_3012 26
#define TRANS4D_3021 27
#define TRANS4D_3102 28
#define TRANS4D_3120 29
#define TRANS4D_3201 30
#define TRANS4D_3210 31


#define DEBUG

#define FCI                             FC::getInstance()

#define MMSHAPE1D(a)                    new std::vector<int>({a})
#define MMSHAPE2D(a, b)                 new std::vector<int>({a, b})
#define MMSHAPE3D(a, b, c)              new std::vector<int>({a, b, c})
#define MMSHAPE4D(a, b, c, d)           new std::vector<int>({a, b, c, d})
#define MMSHAPE5D(a, b, c, d, e)        new std::vector<int>({a, b, c, d, e})
#define MMSHAPE6D(a, b, c, d, e, f)     new std::vector<int>({a, b, c, d, e, f})
#define MMSHAPE7D(a, b, c, d, e, f, g)  new std::vector<int>({a, b, c, d, e, f, g})

#endif // __MACROS_H__
