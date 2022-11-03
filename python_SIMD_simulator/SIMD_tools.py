import numpy as np

def _mm256_setzero_ps():
    pack = 8
    y = np.zeros((pack, )).astype(np.float32)
    return y


def check_pack8(arr, start_i):
    pack = 8
    shape = arr.shape
    assert len(shape) == 1
    numel = shape[0]
    end_i = start_i + pack - 1
    assert start_i >= 0
    assert end_i < numel

def _mm256_loadu_ps(arr, start_i):
    pack = 8
    check_pack8(arr, start_i)
    return arr[start_i:start_i + pack]

def _mm256_storeu_ps(arr, start_i, val):
    pack = 8
    check_pack8(arr, start_i)
    arr[start_i:start_i + pack] = val

def _mm256_broadcast_ss(arr, start_i):
    pack = 8
    shape = arr.shape
    assert len(shape) == 1
    numel = shape[0]
    end_i = start_i
    assert start_i >= 0
    assert end_i < numel
    y = np.zeros((pack, )).astype(np.float32) + arr[start_i]
    return y


def _mm256_fmadd_ps(a, b, c):
    return a * b + c


def _mm256_add_ps(a, b):
    return a + b

def _mm256_mul_ps(a, b):
    return a * b

