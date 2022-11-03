import copy

import numpy as np






# input_name = 'x'
# output_name = 'xaa'


# input_name = 'input'
# output_name = '            V'
# B = np.array([[1, 0, 0, 0],
#               [0, 1, -1, 1],
#               [-1, 1, 1, 0],
#               [0, 0, 0, -1]])



input_name = 'Q'
output_name = '            Y'
B = np.array([[1, 0], [1, 1], [1, -1], [0, -1]])
A = B



# B_T = np.array([[1, 0, -1, 0],
#                 [0, 1, 1, 0],
#                 [0, -1, 1, 0],
#                 [0, -1, 0, 1]])
# return np.matmul(np.matmul(B.T,x),B)
'''
B_T * X = 
           x00 - x20,  x01 - x21,  x02 - x22,  x03 - x23
           x10 + x20,  x11 + x21,  x12 + x22,  x13 + x23
           -x10 + x20,  -x11 + x21,  -x12 + x22,  -x13 + x23
           -x10 + x30,  -x11 + x31,  -x12 + x32,  -x13 + x33

(B_T * X) * B = 
                 x00 - x20 - x02 + x22,  x01 - x21 + x02 - x22, -x01 + x21 + x02 - x22,  -x01 + x21 + x03 - x23, 
                 x10 + x20 -x12 - x22,   x11 + x21 +  x12 + x22,  -x11 - x21 +  x12 + x22,  -x11 - x21 +   x13 + x23, 
                 -x10 + x20 + x12 - x22,  -x11 + x21  -x12 + x22,  x11 - x21  -x12 + x22,  x11 - x21  -x13 + x23, 
                 -x10 + x30 + x12 - x32,  -x11 + x31  -x12 + x32,  x11 - x31  -x12 + x32,  x11 - x31  -x13 + x33, 

'''


# 输出 np.matmul(np.matmul(B.T,x),B) 各元素的表达式
# B.shape = [M, K]
# B_T.shape = [K, M]
# x.shape = [M, M]
# B_T*x.shape = [K, M]
# B_T*x*B.shape = [K, K]



B_T = B.T


M = B.shape[0]
K = B.shape[1]


bt_x = []
bt_x_b = []
for i in range(K):
    aaaaa = []
    for j in range(M):
        aaaaa.append([])
    bt_x.append(aaaaa)

for i in range(K):
    aaaaa = []
    for j in range(K):
        aaaaa.append([])
    bt_x_b.append(aaaaa)

# B_T*x
for i in range(K):
    for j in range(M):
        for k in range(M):
            aaa = B_T[i][k]
            # bbb = '%s[%d][%d]' % (input_name, k, j)
            # bbb = '%s[th+%d][tw+%d][ic]' % (input_name, k, j)
            # bbb = '%s[((th+%d) * pad_W + tw+%d) * in_C + ic]' % (input_name, k, j)
            # bbb = '%s[%d][%d][tid][oc]' % (input_name, k, j)
            bbb = '%s[((%d * tile_size + %d) * tile_num + tid) * out_C + oc]' % (input_name, k, j)
            if aaa == -1:
                bbb = '-' + bbb
            elif aaa == 0:
                bbb = ''
            elif aaa == 1:
                pass
            else:
                exit(1)
            if bbb == '':
                pass
            else:
                bt_x[i][j].append(bbb)


bt_x_clone = np.copy(bt_x)

eles = []

# B_T*x * B
for i in range(K):
    for j in range(K):
        # ele = '%s[%d][%d] = ' % (output_name, i, j)
        # ele = '%s[nn, bb, %d, %d] = ' % (output_name, i, j)
        # ele = '%s[%d, %d, tid, ic] = ' % (output_name, i, j)
        # ele = '%s[((%d * 4 + %d) * tile_num + tid) * in_C + ic] = ' % (output_name, i, j)
        # ele = '%s[%d, %d, tid, oc] = ' % (output_name, i, j)
        ele = '%s[((%d * otile_size + %d) * tile_num + tid) * out_C + oc] = ' % (output_name, i, j)
        for k in range(M):
            aaa = copy.deepcopy(bt_x[i][k])
            bbb = B[k][j]
            if bbb == -1:
                for p in range(len(aaa)):
                    aaa[p] = '-' + aaa[p]
                    aaa[p] = aaa[p].replace('--', '')
            elif bbb == 0:
                aaa = ''
            elif bbb == 1:
                pass
            else:
                exit(1)
            if aaa == '':
                pass
            else:
                for p in range(len(aaa)):
                    bt_x_b[i][j].append(aaa[p])
                    ele += ' + %s' % (aaa[p], )
        eles.append(ele)

eles_new = []
for p in range(len(eles)):
    eee = eles[p]
    eee = eee.replace("=  +", "=")
    eee = eee.replace("+ -", "- ")
    eee = eee.replace("+0]", "]")
    eee = eee + ";"
    eles_new.append(eee)
    print(eles_new[p])

print()







