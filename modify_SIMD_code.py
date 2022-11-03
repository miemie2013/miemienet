import os
import itertools
import copy



def get_args(line):
    # 去掉默认参数
    aa1 = line.split('(')
    aa2 = aa1[1].split(')')
    ss = aa2[0].split(',')
    sss = []
    for s in ss:
        sss.append(s.strip())
    return sss


def mod_code_256_to_128():
    src_path = 'simd_src.txt'
    dst_path = 'simd_dst.txt'
    new_code = ''
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            new_line = copy.deepcopy(line)

            new_line = new_line.replace("__m256", "__m128")
            new_line = new_line.replace("_mm256_loadu_ps", "_mm_load_ps")
            new_line = new_line.replace(" _a", " __a")
            new_line = new_line.replace(" _b", " __b")
            new_line = new_line.replace(" _c", " __c")
            new_line = new_line.replace(" _d", " __d")
            new_line = new_line.replace("(_a", "(__a")
            new_line = new_line.replace("(_b", "(__b")
            new_line = new_line.replace("(_c", "(__c")
            new_line = new_line.replace("(_d", "(__d")
            new_line = new_line.replace("_mm256_broadcast_ss", "_mm_broadcast_ss")
            if '_mm256_fmadd_ps' in new_line:  # 累加
                new_line = new_line.replace("_mm256_fmadd_ps", "_mm_fmadd_ps")
            if '_mm256_storeu_ps' in new_line:  # 保存数据
                new_line = new_line.replace("_mm256_storeu_ps", "_mm_store_ps")
            new_code += new_line
        f.close()
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()

def mod_code_128_to_float():
    src_path = 'simd_src.txt'
    dst_path = 'simd_dst.txt'
    new_code = ''
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            new_line = copy.deepcopy(line)

            # __m128  -->  float
            new_line = new_line.replace("__m128", "float")
            new_line = new_line.replace("_mm_load_ps", "*")
            new_line = new_line.replace("_mm_broadcast_ss", "*")
            if '_mm_fmadd_ps' in new_line:  # 累加
                args = get_args(new_line)
                no_space = new_line.strip()
                new_sent = "%s += %s * %s;" % (args[2], args[0], args[1])
                new_line = new_line.replace(no_space, new_sent)
            if '_mm_store_ps' in new_line:  # 保存数据
                args = get_args(new_line)
                no_space = new_line.strip()
                new_sent = "*%s = %s;" % (args[0], args[1])
                new_line = new_line.replace(no_space, new_sent)
            new_code += new_line
        f.close()
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()


def mod_code2():
    src_path = 'simd_src.txt'
    dst_path = 'simd_dst.txt'
    new_code = ''
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            new_line = copy.deepcopy(line)

            new_line = new_line.replace("__m256", "__m128")
            new_line = new_line.replace("_mm256_broadcast_ss", "_mm_broadcast_ss")
            if '_mm_fmadd_ps' in new_line:  # 累加
                args = get_args(new_line)
                no_space = new_line.strip()
                new_sent = "%s += %s * %s;" % (args[2], args[0], args[1])
                new_line = new_line.replace(no_space, new_sent)
            if '_mm_store_ps' in new_line:  # 保存数据
                args = get_args(new_line)
                no_space = new_line.strip()
                new_sent = "*%s = %s;" % (args[0], args[1])
                new_line = new_line.replace(no_space, new_sent)
            new_code += new_line
        f.close()
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(new_code)
        f.close()
    print()



if __name__ == "__main__":
    # mod_code_256_to_128()
    mod_code_128_to_float()
    # mod_code2()

