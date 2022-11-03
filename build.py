import os
import argparse


def make_parser():
    parser = argparse.ArgumentParser("Build tools.")
    parser.add_argument(
        "--platform", default="LINUX", help="platform, for example: LINUX,WINDOWS,ANDROID"
    )
    parser.add_argument(
        "--cxx", default="g++", help="cxx compiler, for example: g++,cl,nvcc"
    )
    parser.add_argument(
        "--backend", default="", help="backend, for example: BACKEND_X86,BACKEND_ARM"
    )
    parser.add_argument(
        "--OpenCV_INCLUDE_DIRS", default="", help="path to OpenCV_INCLUDE_DIRS"
    )
    parser.add_argument(
        "--OpenCV_LIBS", default="", help="path to OpenCV_LIBS"
    )
    parser.add_argument(
        "--exec_file", default="", help="exec_file"
    )
    return parser

def search_srcs(dir, backend=''):
    srcs = ''
    files = os.listdir(dir)
    for file in files:
        if '.cu' in file or '.cpp' in file:
            if '.cuh' not in file:
                srcs += ' %s' % (dir + '/' + file, )
    return srcs


def build_test(args, platform):
    backend = args.backend
    # for opencv, we use static lib
    include_directories = ' -I miemienet -I miemiedet '
    if platform == 'LINUX':
        opencv_lib = ''
    elif platform == 'WINDOWS':
        include_directories += ' -I \"%s\"' % (args.OpenCV_INCLUDE_DIRS,)
        opencv_lib = ' -l opencv_world455 -L \"%s\"' % (args.OpenCV_LIBS, )

    # miemienet
    miemienet_srcs = search_srcs('miemienet')
    framework_srcs = search_srcs('miemienet/framework', args.backend)
    nn_srcs = search_srcs('miemienet/nn')
    backend_srcs = ''
    if args.backend == 'BACKEND_X86':
        backend_srcs = search_srcs('miemienet/nn/common')
    if args.backend == 'BACKEND_ARM':
        backend_srcs = search_srcs('miemienet/nn/common')

    # miemiedet
    miemiedet_models_architectures_srcs = search_srcs('miemiedet/models/architectures')
    miemiedet_models_backbones_srcs = search_srcs('miemiedet/models/backbones')
    miemiedet_models_necks_srcs = search_srcs('miemiedet/models/necks')
    miemiedet_models_heads_srcs = search_srcs('miemiedet/models/heads')

    exec_file = args.exec_file
    main_src = 'test/%s.cpp' % exec_file

    if platform == 'LINUX':
        exec_file += '.out'
    cmd = '%s' % args.cxx
    cmd += ' -D%s' % platform
    cmd += ' -D%s' % args.backend
    cmd += ' %s' % include_directories
    cmd += ' %s' % opencv_lib
    cmd += ' %s' % miemienet_srcs
    cmd += ' %s' % framework_srcs
    cmd += ' %s' % nn_srcs
    cmd += ' %s' % backend_srcs
    cmd += ' %s' % miemiedet_models_architectures_srcs
    cmd += ' %s' % miemiedet_models_backbones_srcs
    cmd += ' %s' % miemiedet_models_necks_srcs
    cmd += ' %s' % miemiedet_models_heads_srcs
    cmd += ' %s' % main_src
    if backend == 'BACKEND_X86':
        cmd += ' -fopenmp'   # 为了使用 #pragma omp parallel for num_threads(12)
        cmd += ' -march=native'   # 为了使用 AVX SIMD指令
    cmd += ' -o'
    cmd += ' %s' % exec_file
    if platform == 'LINUX':   # LINUX g++编译器下，不打印warning
        cmd += ' -w'
        # g++编译器3级优化。实测对于matmul_x86.cpp，矩阵乘法得到极大加速。
        # 甚至，把 matmul_common.cpp 优化到比 matmul_x86.cpp(使用了SIMD指令) 快一点。
        # 注意，如果使用-Ofast，会导致reduce mean op结果出现nan，使用-O3优化则不会。
        cmd += ' -O3'   # g++编译器3级优化。
        cmd += ' `pkg-config --cflags --libs opencv`'
    print(cmd)
    result = os.system(cmd)
    print(result)


if __name__ == "__main__":
    args = make_parser().parse_args()
    platform = args.platform
    assert platform in ['LINUX', 'WINDOWS']
    assert args.backend in ['BACKEND_X86', 'BACKEND_ARM']

    cxx_compiler = ''
    if platform == 'LINUX':
        cxx_compiler = 'g++'
    elif platform == 'WINDOWS':
        cxx_compiler = 'cl'

    # ------------------------------- for -------------------------------
    # main_src = 'examples/for.cpp'
    # exec_file = 'for'
    # if platform == 'LINUX':
    #     exec_file += '.out'
    # cmd = '%s %s -o %s' % (cxx_compiler, main_src, exec_file)
    # result = os.system(cmd)
    # print(result)

    build_test(args, platform)

