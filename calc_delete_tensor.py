
import numpy as np


if __name__ == "__main__":
    '''
    验证创建和释放的张量数是否相同
    '''

    cre_count = 0
    del_count = 0
    cre_ids = []
    del_ids = []
    with open('del.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 'delete Tensor' in line:
                del_count += 1
                ss = line.split('id=')
                sss = ss[1].split(', dims=')
                del_ids.append(int(sss[0]))
                # print(line)
            if 'create Tensor' in line:
                cre_count += 1
                ss = line.split('id=')
                sss = ss[1].split(', dims=')
                cre_ids.append(int(sss[0]))
    print(cre_count)
    print(del_count)
    cre_ids = np.array(cre_ids)
    del_ids = np.array(del_ids)
    cre_ids = np.sort(cre_ids)
    del_ids = np.sort(del_ids)
    print(cre_ids)
    print(del_ids)
    not_delete_ids = []
    for idd in cre_ids:
        if idd not in del_ids:
            not_delete_ids.append(idd)
    print(not_delete_ids)
    pass

