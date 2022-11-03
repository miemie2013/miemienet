import os
import itertools
import copy


def gen_transpose_common():
    # 打印全排列，transpose op用到
    trans2d = list(itertools.permutations([0, 1]))
    trans3d = list(itertools.permutations([0, 1, 2]))
    trans4d = list(itertools.permutations([0, 1, 2, 3]))
    for trans2d_e in trans2d:
        print("%d%d" % (trans2d_e[0], trans2d_e[1]))
    print()
    for trans3d_e in trans3d:
        print("%d%d%d" % (trans3d_e[0], trans3d_e[1], trans3d_e[2]))
    print()
    for trans4d_e in trans4d:
        print("%d%d%d%d" % (trans4d_e[0], trans4d_e[1], trans4d_e[2], trans4d_e[3]))
    print()
    content_cpp = ''
    content_cpp_invoke = ''
    content_forward = ''

    # 2d
    D = ['H', 'W']
    d = ['h', 'w']
    kkk = 0
    for t in trans2d:
        template = 'template<typename data_t>\n' + \
                   'void transpose2d_%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s) {\n' % (t[0], t[1], D[0], D[1]) + \
                   '    // y[%s][%s] = x[%s][%s]\n' % (d[t[0]], d[t[1]], d[0], d[1]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            y[(%s * %s) + %s] = x[(%s * %s) + %s];\n' % (d[t[0]], D[t[1]], d[t[1]],      d[0], D[1], d[1]) + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS2D_%d%d) {\n' % (t[0], t[1]) + \
                   '            if (input->dims != %d) {\n' % (2, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS2D_%d%d, input->dims != %d\\n");\n' % (t[0], t[1], 2) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            transpose2d_%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s);\n' % (t[0], t[1], D[0], D[1]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS2D_%d%d) {\n' % (t[0], t[1]) + \
                   '        if (input->dims != %d) {\n' % (2, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS2D_%d%d, input->dims != %d\\n");\n' % (t[0], t[1], 2) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        output = new SNT Tensor(MMSHAPE2D(%s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]]) + \
                   '    }\n'
        if kkk == 0:
            templat2 = templat2.replace("else if (", "if (")
            templat3 = templat3.replace("else if (", "if (")
        kkk += 1
        content_cpp_invoke += templat2
        content_forward += templat3

    # 3d
    D = ['N', 'H', 'W']
    d = ['n', 'h', 'w']
    for t in trans3d:
        template = 'template<typename data_t>\n' + \
                   'void transpose3d_%d%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s, int %s) {\n' % (t[0], t[1], t[2], D[0], D[1], D[2]) + \
                   '    // y[%s][%s][%s] = x[%s][%s][%s]\n' % (d[t[0]], d[t[1]], d[t[2]], d[0], d[1], d[2]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            for (int %s = 0; %s < %s; %s++) {\n' % (d[2], d[2], D[2], d[2]) + \
                   '                y[((%s * %s) + %s) * %s + %s] = x[((%s * %s) + %s) * %s + %s];\n' % (d[t[0]], D[t[1]], d[t[1]], D[t[2]], d[t[2]],      d[0], D[1], d[1], D[2], d[2]) + \
                   '            }\n' + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS3D_%d%d%d) {\n' % (t[0], t[1], t[2]) + \
                   '            if (input->dims != %d) {\n' % (3, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS3D_%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], 3) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '            transpose3d_%d%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s, %s);\n' % (t[0], t[1], t[2], D[0], D[1], D[2]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS3D_%d%d%d) {\n' % (t[0], t[1], t[2]) + \
                   '        if (input->dims != %d) {\n' % (3, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS3D_%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], 3) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '        output = new SNT Tensor(MMSHAPE3D(%s, %s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]], D[t[2]]) + \
                   '    }\n'
        content_cpp_invoke += templat2
        content_forward += templat3

    # 4d
    D = ['N', 'C', 'H', 'W']
    d = ['n', 'c', 'h', 'w']
    for t in trans4d:
        template = 'template<typename data_t>\n' + \
                   'void transpose4d_%d%d%d%d_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int %s, int %s, int %s, int %s) {\n' % (t[0], t[1], t[2], t[3], D[0], D[1], D[2], D[3]) + \
                   '    // y[%s][%s][%s][%s] = x[%s][%s][%s][%s]\n' % (d[t[0]], d[t[1]], d[t[2]], d[t[3]], d[0], d[1], d[2], d[3]) + \
                   '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                   '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                   '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                   '            for (int %s = 0; %s < %s; %s++) {\n' % (d[2], d[2], D[2], d[2]) + \
                   '                for (int %s = 0; %s < %s; %s++) {\n' % (d[3], d[3], D[3], d[3]) + \
                   '                    y[(((%s * %s) + %s) * %s + %s) * %s + %s] = x[(((%s * %s) + %s) * %s + %s) * %s + %s];\n' % (d[t[0]], D[t[1]], d[t[1]], D[t[2]], d[t[2]], D[t[3]], d[t[3]],      d[0], D[1], d[1], D[2], d[2], D[3], d[3]) + \
                   '                }\n' + \
                   '            }\n' + \
                   '        }\n' + \
                   '    }\n' + \
                   '}\n'
        content_cpp += template + '\n'
        templat2 = '        else if (transpose_type == TRANS4D_%d%d%d%d) {\n' % (t[0], t[1], t[2], t[3]) + \
                   '            if (input->dims != %d) {\n' % (4, ) + \
                   '                printf("Error from transpose op, transpose_type == TRANS4D_%d%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], t[3], 4) + \
                   '                exit(1);\n' + \
                   '            }\n' + \
                   '            const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '            const int %s = input->shape->at(%d);\n' % (D[3], 3) + \
                   '            transpose4d_%d%d%d%d_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, %s, %s, %s, %s);\n' % (t[0], t[1], t[2], t[3], D[0], D[1], D[2], D[3]) + \
                   '        }\n'
        templat3 = '    else if (transpose_type == TRANS4D_%d%d%d%d) {\n' % (t[0], t[1], t[2], t[3]) + \
                   '        if (input->dims != %d) {\n' % (4, ) + \
                   '            printf("Error from transpose op, transpose_type == TRANS4D_%d%d%d%d, input->dims != %d\\n");\n' % (t[0], t[1], t[2], t[3], 4) + \
                   '            exit(1);\n' + \
                   '        }\n' + \
                   '        const int %s = input->shape->at(%d);\n' % (D[0], 0) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[1], 1) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[2], 2) + \
                   '        const int %s = input->shape->at(%d);\n' % (D[3], 3) + \
                   '        output = new SNT Tensor(MMSHAPE4D(%s, %s, %s, %s), FP32, false, false);\n' % (D[t[0]], D[t[1]], D[t[2]], D[t[3]]) + \
                   '    }\n'
        content_cpp_invoke += templat2
        content_forward += templat3

    with open('gen_code_cpp.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp)
        f.close()
    with open('gen_code_cpp_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp_invoke)
        f.close()


    content_x86 = content_cpp.replace("_cpp_kernel(", "_x86_kernel(") + '\n'
    content_x86_invoke = content_cpp_invoke.replace("_cpp_kernel<", "_x86_kernel<")
    content_x86 = '#if BACKEND_X86\n' + content_x86 + '#endif // BACKEND_X86\n'
    content_x86_invoke = '#if BACKEND_X86\n' + content_x86_invoke + '#endif // BACKEND_X86\n'
    with open('gen_code_x86.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86)
        f.close()
    with open('gen_code_x86_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86_invoke)
        f.close()

    # 直接替换代码
    '''
注解对配对使用：
// gen cpp code start
// gen cpp code end

// gen x86 code start
// gen x86 code end

// gen cpp invoke code start
// gen cpp invoke code end

// gen x86 invoke code start
// gen x86 invoke code end

// gen forward code start
// gen forward code end

    '''
    src_path = 'miemienet/nn/common/transpose_common.cpp'
    new_code = ''
    paste_zone = False
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if paste_zone:
                if 'gen cpp code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen cpp invoke code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 invoke code end' in line:
                    paste_zone = False
                    new_code += line
            else:
                new_code += line
                if 'gen cpp code start' in line:
                    paste_zone = True
                    new_code += content_cpp
                if 'gen x86 code start' in line:
                    paste_zone = True
                    new_code += content_x86
                if 'gen cpp invoke code start' in line:
                    paste_zone = True
                    new_code += content_cpp_invoke
                if 'gen x86 invoke code start' in line:
                    paste_zone = True
                    new_code += content_x86_invoke
        f.close()
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()

    src_path = 'miemienet/nn/transpose.cpp'
    new_code = ''
    paste_zone = False
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if paste_zone:
                if 'gen forward code end' in line:
                    paste_zone = False
                    new_code += line
            else:
                new_code += line
                if 'gen forward code start' in line:
                    paste_zone = True
                    new_code += content_forward
        f.close()
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()

def elem_get_index(d, D, shape):
    real_shape = []
    real_i = []
    p = 0
    for s in shape:
        if s != '1':
            real_shape.append(D[p])
            real_i.append(d[p])
        p += 1
    if len(real_shape) == 4:
        _index = '((%s * %s + %s) * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2], real_shape[3], real_i[3])
    elif len(real_shape) == 3:
        _index = '(%s * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2])
    elif len(real_shape) == 2:
        _index = '%s * %s + %s' % (real_i[0], real_shape[1], real_i[1])
    elif len(real_shape) == 1:
        _index = '%s' % (real_i[0], )
    elif len(real_shape) == 0:
        _index = '0'
    return _index

def elem_get_out_index(d, D, shape1, shape2):
    real_shape = []
    real_i = []
    p = 0
    for s1 in shape1:
        s2 = shape2[p]
        if s1 != '1' or s2 != '1':
            real_shape.append(D[p])
            real_i.append(d[p])
        p += 1

    if len(real_shape) == 4:
        _index = '((%s * %s + %s) * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2], real_shape[3], real_i[3])
    elif len(real_shape) == 3:
        _index = '(%s * %s + %s) * %s + %s' % (real_i[0], real_shape[1], real_i[1], real_shape[2], real_i[2])
    elif len(real_shape) == 2:
        _index = '%s * %s + %s' % (real_i[0], real_shape[1], real_i[1])
    elif len(real_shape) == 1:
        _index = '%s' % (real_i[0], )
    elif len(real_shape) == 0:
        _index = '0'
    return _index

def gen_elementwise_common():
    ndim = 4
    D = ['N', 'C', 'H', 'W']
    d = ['n', 'c', 'h', 'w']
    tensor1_shapes = []
    tensor2_shapes = []
    for i in range(ndim + 1):
        tensor1_shape = list(itertools.combinations(range(ndim), i))
        for shape1 in tensor1_shape:
            cp1 = copy.deepcopy(D)
            for i1 in shape1:
                cp1[i1] = '1'
            tensor1_shapes.append(cp1)
            tensor2_shapes.append(copy.deepcopy(cp1))

    content_cpp_xop = ''
    content_cpp = ''
    content_cpp_invoke  = '        const int N0 = a->shape->at(0);\n'
    content_cpp_invoke += '        const int C0 = a->shape->at(1);\n'
    content_cpp_invoke += '        const int H0 = a->shape->at(2);\n'
    content_cpp_invoke += '        const int W0 = a->shape->at(3);\n'
    content_cpp_invoke += '        const int N1 = b->shape->at(0);\n'
    content_cpp_invoke += '        const int C1 = b->shape->at(1);\n'
    content_cpp_invoke += '        const int H1 = b->shape->at(2);\n'
    content_cpp_invoke += '        const int W1 = b->shape->at(3);\n'
    content_cpp_invoke += '        const int N = std::max(N0, N1);\n'
    content_cpp_invoke += '        const int C = std::max(C0, C1);\n'
    content_cpp_invoke += '        const int H = std::max(H0, H1);\n'
    content_cpp_invoke += '        const int W = std::max(W0, W1);\n'
    content_cpp_xop_invoke = ''
    kkk = 0
    for s1 in tensor1_shapes:
        for s2 in tensor2_shapes:
            # 是否过滤掉一些不常见情况。过滤掉的话可以减少代码量，提高编译速度。
            filt_ = False
            filt_ = True
            continue_ = True
            if filt_:
                # 添加白名单。白名单不会跳过
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == 'N' and s2[1] == 'C' and s2[2] == 'H' and s2[3] == 'W':
                    continue_ = False
                if s1[0] == '1' and s1[1] == '1' and s1[2] == '1' and s2[0] == '1' and s2[1] == '1' and s2[2] == '1':
                    continue_ = False
                if s1[0] == '1' and s1[1] == '1' and s2[0] == '1' and s2[1] == '1':
                    continue_ = False
                if s1[0] == '1' and s2[0] == '1':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == '1' and s2[1] == '1' and s2[2] == '1' and s2[3] == 'W':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == 'C' and s1[2] == 'H' and s1[3] == 'W' and s2[0] == '1' and s2[1] == 'C' and s2[2] == '1' and s2[3] == '1':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == '1' and s1[2] == '1' and s1[3] == 'W' and s2[0] == 'N' and s2[1] == '1' and s2[2] == '1' and s2[3] == 'W':
                    continue_ = False
                if s1[0] == 'N' and s1[1] == '1' and s1[2] == '1' and s1[3] == 'W' and s2[0] == '1' and s2[1] == '1' and s2[2] == '1' and s2[3] == 'W':
                    continue_ = False
            if filt_ and continue_:
                continue
            _left = 'x[%s]' % (elem_get_index(d, D, s1))
            _right = 'y[%s]' % (elem_get_index(d, D, s2))
            _out = 'z[%s]' % (elem_get_out_index(d, D, s1, s2))
            template = 'template<typename data_t>\n' + \
                       'void elem4d_%s%s%s%s_oopp_%s%s%s%s_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int %s, int %s, int %s, int %s) {\n' % (s1[0], s1[1], s1[2], s1[3], s2[0], s2[1], s2[2], s2[3], D[0], D[1], D[2], D[3]) + \
                       '    #pragma omp parallel for num_threads(num_threads_)\n' + \
                       '    for (int %s = 0; %s < %s; %s++) {\n' % (d[0], d[0], D[0], d[0]) + \
                       '        for (int %s = 0; %s < %s; %s++) {\n' % (d[1], d[1], D[1], d[1]) + \
                       '            for (int %s = 0; %s < %s; %s++) {\n' % (d[2], d[2], D[2], d[2]) + \
                       '                for (int %s = 0; %s < %s; %s++) {\n' % (d[3], d[3], D[3], d[3]) + \
                       '                    %s = qwel%sqwem%sqwer;\n' % (_out, _left, _right) + \
                       '                }\n' + \
                       '            }\n' + \
                       '        }\n' + \
                       '    }\n' + \
                       '}\n'
            content_cpp_xop += template + '\n'
            cond1 = ''
            if s1[0] == D[0] and s2[0] == D[0]:
                cond1 = 'N0 == N1 && N1 > 1'
            if s1[0] == D[0] and s2[0] == '1':
                cond1 = 'N0 > 1 && N1 == 1'
            if s1[0] == '1' and s2[0] == D[0]:
                cond1 = 'N0 == 1 && N1 > 1'
            if s1[0] == '1' and s2[0] == '1':
                cond1 = 'N0 == 1 && N1 == 1'
            cond2 = ''
            if s1[1] == D[1] and s2[1] == D[1]:
                cond2 = 'C0 == C1 && C1 > 1'
            if s1[1] == D[1] and s2[1] == '1':
                cond2 = 'C0 > 1 && C1 == 1'
            if s1[1] == '1' and s2[1] == D[1]:
                cond2 = 'C0 == 1 && C1 > 1'
            if s1[1] == '1' and s2[1] == '1':
                cond2 = 'C0 == 1 && C1 == 1'
            cond3 = ''
            if s1[2] == D[2] and s2[2] == D[2]:
                cond3 = 'H0 == H1 && H1 > 1'
            if s1[2] == D[2] and s2[2] == '1':
                cond3 = 'H0 > 1 && H1 == 1'
            if s1[2] == '1' and s2[2] == D[2]:
                cond3 = 'H0 == 1 && H1 > 1'
            if s1[2] == '1' and s2[2] == '1':
                cond3 = 'H0 == 1 && H1 == 1'
            cond4 = ''
            if s1[3] == D[3] and s2[3] == D[3]:
                cond4 = 'W0 == W1 && W1 > 1'
            if s1[3] == D[3] and s2[3] == '1':
                cond4 = 'W0 > 1 && W1 == 1'
            if s1[3] == '1' and s2[3] == D[3]:
                cond4 = 'W0 == 1 && W1 > 1'
            if s1[3] == '1' and s2[3] == '1':
                cond4 = 'W0 == 1 && W1 == 1'
            templat2 = '            else if (%s && %s && %s && %s) {\n' % (cond1, cond2, cond3, cond4) + \
                       '                elem4d_%s%s%s%s_oopp_%s%s%s%s_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, %s, %s, %s, %s);\n' % (s1[0], s1[1], s1[2], s1[3], s2[0], s2[1], s2[2], s2[3], D[0], D[1], D[2], D[3]) + \
                       '            }\n'
            if kkk == 0:
                templat2 = templat2.replace("else if (", "if (")
            kkk += 1
            content_cpp_xop_invoke += templat2
    # 不支持的情况错误提醒
    content_cpp_xop_invoke += '            else {\n' + \
                              '                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\\n", N0, C0, H0, W0, N1, C1, H1, W1);\n' + \
                              '                exit(1);\n' + \
                              '            }\n'
    content_cpp_invoke += '        if (op_type == %s) {\n' % ('ELE_ADD', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'add')
    content_cpp_invoke += '        }\n'
    content_cpp_invoke += '        else if (op_type == %s) {\n' % ('ELE_SUB', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'sub')
    content_cpp_invoke += '        }\n'
    content_cpp_invoke += '        else if (op_type == %s) {\n' % ('ELE_MUL', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'mul')
    content_cpp_invoke += '        }\n'
    content_cpp_invoke += '        else if (op_type == %s) {\n' % ('ELE_DIV', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'div')
    content_cpp_invoke += '        }\n'
    content_cpp_invoke += '        else if (op_type == %s) {\n' % ('ELE_MIN', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'min')
    content_cpp_invoke += '        }\n'
    content_cpp_invoke += '        else if (op_type == %s) {\n' % ('ELE_MAX', )
    content_cpp_invoke += content_cpp_xop_invoke.replace('oopp', 'max')
    content_cpp_invoke += '        }\n'
    content_cpp += content_cpp_xop.replace('oopp', 'add').replace('qwel', '').replace('qwem', ' + ').replace('qwer', '')
    content_cpp += content_cpp_xop.replace('oopp', 'sub').replace('qwel', '').replace('qwem', ' - ').replace('qwer', '')
    content_cpp += content_cpp_xop.replace('oopp', 'mul').replace('qwel', '').replace('qwem', ' * ').replace('qwer', '')
    content_cpp += content_cpp_xop.replace('oopp', 'div').replace('qwel', '').replace('qwem', ' / ').replace('qwer', '')
    content_cpp += content_cpp_xop.replace('oopp', 'min').replace('qwel', 'std::min(').replace('qwem', ', ').replace('qwer', ')')
    content_cpp += content_cpp_xop.replace('oopp', 'max').replace('qwel', 'std::max(').replace('qwem', ', ').replace('qwer', ')')



    with open('gen_code_cpp.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp)
        f.close()
    with open('gen_code_cpp_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_cpp_invoke)
        f.close()

    content_x86 = content_cpp.replace("_cpp_kernel(", "_x86_kernel(") + '\n'
    content_x86_invoke = content_cpp_invoke.replace("_cpp_kernel<", "_x86_kernel<")
    content_x86 = '#if BACKEND_X86\n' + content_x86 + '#endif // BACKEND_X86\n'
    content_x86_invoke = '#if BACKEND_X86\n' + content_x86_invoke + '#endif // BACKEND_X86\n'
    with open('gen_code_x86.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86)
        f.close()
    with open('gen_code_x86_invoke.txt', 'w', encoding='utf-8') as f:
        f.write(content_x86_invoke)
        f.close()

    # 直接替换代码
    src_path = 'miemienet/nn/common/elementwise_common.cpp'
    new_code = ''
    paste_zone = False
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if paste_zone:
                if 'gen cpp code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen cpp invoke code end' in line:
                    paste_zone = False
                    new_code += line
                if 'gen x86 invoke code end' in line:
                    paste_zone = False
                    new_code += line
            else:
                new_code += line
                if 'gen cpp code start' in line:
                    paste_zone = True
                    new_code += content_cpp
                if 'gen x86 code start' in line:
                    paste_zone = True
                    new_code += content_x86
                if 'gen cpp invoke code start' in line:
                    paste_zone = True
                    new_code += content_cpp_invoke
                if 'gen x86 invoke code start' in line:
                    paste_zone = True
                    new_code += content_x86_invoke
        f.close()
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()


def get_construct_args(args):
    # 去掉默认参数
    assert args[0] == '('
    assert args[-1] == ')'
    ss = args[1:-1].split(',')
    new_args = []
    for s in ss:
        s = s.strip()
        if 'Tensor* ' in s:
            continue
        if 'std::vector<Tensor*>* ' in s:
            continue
        new_args.append(s)
    result = ''
    for s in new_args:
        result += s + ', '
    result = '(' + result[:-2] + ')'
    return result

def args_no_default(args):
    # 去掉默认参数
    assert args[0] == '('
    assert args[-1] == ')'
    ss = args[1:-1].split(',')
    new_args = []
    for s in ss:
        s = s.strip()
        if '=' in s:
            new_args.append(s.split('=')[0])
        else:
            new_args.append(s)
    result = ''
    for s in new_args:
        result += s + ', '
    result = '(' + result[:-2] + ')'
    return result

def args_no_type(args):
    # 去掉类型
    assert args[0] == '('
    assert args[-1] == ')'
    ss = args[1:-1].split(',')
    new_args = []
    for s in ss:
        s = s.strip()
        if '* ' in s:
            new_args.append(s.split('* ')[1])
        elif ' *' in s:
            new_args.append(s.split(' *')[1])
        elif ' ' in s:
            new_args.append(s.split(' ')[1])
        elif '  ' in s:
            new_args.append(s.split('  ')[1])
        else:
            raise NotImplemented("")
    result = ''
    for s in new_args:
        result += s + ', '
    result = '(' + result[:-2] + ')'
    return result

def get_chengyuan(args):
    # 获取层的成员变量
    assert args[0] == '('
    assert args[-1] == ')'
    ss = args[1:-1].split(',')
    result = []
    for s in ss:
        s = s.strip()
        result.append(s)
    return result

def gen_new_op():
    # F_args 指的是 xxx_common.h 里函数申明的参数列表
    # construct_args 比 F_args，少了 Tensor* , std::vector<Tensor*>* 等参数
    op_name = 'avgpool2d'
    class_name = 'AvgPool2d'
    F_args = '(Tensor* input, Tensor* output, int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, bool ceil_mode=false)'
    forward_type = 'SISO'

    op_name = 'activation'
    class_name = 'Activation'
    F_args = '(Tensor* input, Tensor* output, char* type, float alpha)'
    forward_type = 'SISO'

    op_name = 'softmax'
    class_name = 'Softmax'
    F_args = '(Tensor* input, Tensor* output, int dim=-1)'
    forward_type = 'SISO'

    op_name = 'concat'
    class_name = 'Concat'
    F_args = '(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4, Tensor* output, int dim=-1)'
    forward_type = 'MISO'

    op_name = 'interp'
    class_name = 'Interp'
    F_args = '(Tensor* input, Tensor* output, int size_h=0, int size_w=0, float scale_h=-1.f, float scale_w=-1.f, char* mode="nearest", bool align_corners=false, bool recompute_scale_factor=false)'
    forward_type = 'SISO'

    op_name = 'reduce'
    class_name = 'Reduce'
    F_args = '(Tensor* input, Tensor* output, std::vector<int>* dims, bool keepdim, int op_type)'
    forward_type = 'SISO'

    op_name = 'transpose'
    class_name = 'Transpose'
    F_args = '(Tensor* input, Tensor* output, int transpose_type)'
    forward_type = 'SISO'

    construct_args = get_construct_args(F_args)

    print("modify miemienet.h :")
    aaa = '#include "nn/%s.h"' % (op_name, )
    print(aaa)
    aaa = '#include "nn/common/%s_common.h"' % (op_name, )
    print(aaa)
    print()

    nn_h_name = 'miemienet/nn/%s.h' % (op_name, )
    if not os.path.exists(nn_h_name):
        with open(nn_h_name, 'w', encoding='utf-8') as f:
            f.write('')
            f.close()
    print("new file %s :" % nn_h_name)
    nn_h_macro = '__%s_H__' % (op_name.upper(), )
    print("macro: %s" % nn_h_macro)
    construct_args1 = args_no_default(construct_args)
    chengyuan = get_chengyuan(construct_args1)
    for cy in chengyuan:
        print('    %s;' % cy)
    print("class name: %s%s" % (class_name, construct_args))
    print()

    nn_cpp_name = 'miemienet/nn/%s.cpp' % (op_name, )
    if not os.path.exists(nn_cpp_name):
        with open(nn_cpp_name, 'w', encoding='utf-8') as f:
            f.write('')
            f.close()
    print("new file %s :" % nn_cpp_name)
    print('#include "%s.h"' % op_name)
    print('#include "common/%s_common.h"' % op_name)
    print("class name: %s%s" % (class_name, construct_args1))
    print()
    for cy in chengyuan:
        cyy = cy.split(' ')[1]
        print('    this->%s = %s;' % (cyy, cyy))
    F_args1 = args_no_default(F_args)
    F_args2 = args_no_type(F_args1)
    print("    miemienet::functional::%s%s;" % (op_name, F_args2))
    print()

    nn_common_h_name = 'miemienet/nn/common/%s_common.h' % (op_name, )
    if not os.path.exists(nn_common_h_name):
        with open(nn_common_h_name, 'w', encoding='utf-8') as f:
            f.write('')
            f.close()
    print("new file %s :" % nn_common_h_name)
    nn_common_h_macro = '__F_%s_COMMON_H__' % (op_name.upper(), )
    print("macro: %s" % nn_common_h_macro)
    if forward_type == "SISO":
        print('void %s%s;' % (op_name, F_args)); print();
    elif forward_type == "MISO":
        print('void %s%s;' % (op_name, F_args)); print();
    print()

    nn_common_cpp_name = 'miemienet/nn/common/%s_common.cpp' % (op_name, )
    if not os.path.exists(nn_common_cpp_name):
        with open(nn_common_cpp_name, 'w', encoding='utf-8') as f:
            f.write('')
            f.close()
    print("new file %s :" % nn_common_cpp_name)
    if forward_type == "SISO":
        print('void %s%s' % (op_name, F_args1)); print();
    elif forward_type == "MISO":
        print('void %s%s' % (op_name, F_args1)); print();
    print()


def gen_miemiedet_code():
    class_name = 'ConvBNLayer'
    construct_args = '(int ch_in, int ch_out, int filter_size=3, int stride=1, int groups=1, int padding=0, char* act_name=nullptr)'
    forward_type = 'SISO'

    class_name = 'RepVggBlock'
    construct_args = '(int ch_in, int ch_out, char* act_name="relu")'
    forward_type = 'SISO'

    class_name = 'BasicBlock'
    construct_args = '(int ch_in, int ch_out, char* act_name="relu", bool shortcut=true)'
    forward_type = 'SISO'

    class_name = 'EffectiveSELayer'
    construct_args = '(int channels, char* act_name="hardsigmoid")'
    forward_type = 'SISO'

    class_name = 'CSPResStage'
    construct_args = '(int ch_in, int ch_out, int n, int stride, char* act_name="relu", bool use_attn=true)'
    forward_type = 'SISO'

    class_name = 'CSPResNet'
    construct_args = '(std::vector<int>* layers, std::vector<int>* channels, char* act_name="swish", std::vector<int>* return_idx, bool depth_wise=false, bool use_large_stem=false, float width_mult=1.f, float depth_mult=1.f, int freeze_at=-1)'
    forward_type = 'SIMO'

    class_name = 'SPP'
    construct_args = '(int ch_in, int ch_out, int k, char* act_name="swish")'
    forward_type = 'SISO'

    class_name = 'CSPStage'
    construct_args = '(int ch_in, int ch_out, int n, char* act_name="swish", bool spp=false)'
    forward_type = 'SISO'

    class_name = 'CustomCSPPAN'
    construct_args = '(std::vector<int>* in_channels, std::vector<int>* out_channels, char* act_name="leakyrelu", int stage_num=1, int block_num=3, bool drop_block=false, int block_size=3, float keep_prob=0.9f, bool spp=false, float width_mult=1.f, float depth_mult=1.f)'
    forward_type = 'MIMO'

    class_name = 'ESEAttn'
    construct_args = '(int feat_channels, char* act_name="swish")'
    forward_type = 'MISO'

    class_name = 'PPYOLOEHead'
    construct_args = '(std::vector<int>* in_channels, int num_classes=80, char* act_name="swish", std::vector<float>* fpn_strides=nullptr, float grid_cell_scale=5.f, float grid_cell_offset=0.5f, int reg_max=16, int static_assigner_epoch=4, bool use_varifocal_loss=true)'
    forward_type = 'MIMO'

    class_name = 'PicoHeadV2'
    construct_args = '(std::vector<int>* in_channels, int num_classes=80, std::vector<float>* fpn_stride=nullptr, bool use_align_head=true, int reg_max=16, int feat_in_chan=96, float cell_offset=0.f, char* act_name="hardswish", float grid_cell_scale=5.f)'
    forward_type = 'MIMO'

    print("new file *.h :")
    construct_args1 = args_no_default(construct_args)
    construct_args2 = args_no_type(construct_args1)
    chengyuan = get_chengyuan(construct_args1)
    print("class name: %s%s" % (class_name, construct_args))
    for cy in chengyuan:
        print('    %s;' % cy)
    print()

    print("new file *.cpp :")
    print("class name: %s%s" % (class_name, construct_args1))
    # print('    this->%s = %s;' % ("forward_type", forward_type))
    for cy in chengyuan:
        cyy = cy.split(' ')[1]
        print('    this->%s = %s;' % (cyy, cyy))
    print(construct_args2)



if __name__ == "__main__":
    # 生成新op的需要增加的代码
    # gen_new_op()
    # 根据构造函数的参数列表，生成C++的成员变量声明，以及 this->xxx = xxx;语句。
    gen_miemiedet_code()

    # 生成 transpose op，common 版本的代码
    # gen_transpose_common()
    # 生成 elementwise op，common 版本的代码
    # gen_elementwise_common()

