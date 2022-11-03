import numpy as np
from SIMD_tools import _mm256_setzero_ps, _mm256_loadu_ps, _mm256_fmadd_ps, _mm256_broadcast_ss


'''
用python模拟SIMD的矩阵乘法。
SIMD的寄存器，每次

'''
zeros =  [1., 2., 3., 3., 7., 8., 5., 6.]
zeros2 = [1., 2., 3., 3., 7., 8., 5., 6.]
zeros3 = [1., 0., 0., 0., 0., 0., 0., 0.]
zeros4 = [1., 0., 0., 0., 0., 0., 0., 0.]

zeros = np.array(zeros).astype(np.float32)
zeros2 = np.array(zeros2).astype(np.float32)
zeros3 = np.array(zeros3).astype(np.float32)
zeros4 = np.array(zeros4).astype(np.float32)

oc = 0
out_elempack = 8


_sum = _mm256_setzero_ps()
_v1 = _mm256_loadu_ps(zeros, oc * out_elempack)
_v2 = _mm256_loadu_ps(zeros2, oc * out_elempack)

_sum = _mm256_fmadd_ps(_v1, _v2, _sum)
print(_sum)
print()

_v3 = _mm256_loadu_ps(zeros3, oc * out_elempack)
_v4 = _mm256_loadu_ps(zeros4, oc * out_elempack)
_sum = _mm256_fmadd_ps(_v3, _v4, _sum)
print(_sum)


# 测试广播
bbb = _mm256_broadcast_ss(zeros4, 0)


print()







