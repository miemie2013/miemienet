import copy
import struct
import numpy as np
import math





def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            print(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model





def save_as_txt(name, tensor):
    ndarray = tensor.cpu().detach().numpy()
    save_ndarray_as_txt(name, ndarray)


def save_ndarray_as_txt(name, ndarray):
    content = ''
    array_flatten = np.copy(ndarray)
    array_flatten = np.reshape(array_flatten, (-1, ))
    n = array_flatten.shape[0]
    for i in range(n):
        content += '%f\n' % array_flatten[i]
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()


def save_weights_as_txt(name, model_std):
    param_names = model_std.keys()
    for param_name in param_names:
        if '-' in param_name:
            print('\'-\' cant appear in param_name!')
            exit(0)
        tensor = model_std[param_name]
        save_as_txt("%s-%s.txt" % (name, param_name), tensor)



def bp_write_value(bp, value, force_fp32=True):
    if force_fp32:
        s = struct.pack('f', value)
    else:
        raise NotImplementedError("fp16 is not implemented.")
    bp.write(s)
    return bp


def exist_key(param_names, target):
    for param_name in param_names:
        if target == param_name:
            return True
    return False

def read_weights_from_miemienet(name, ndarray):
    shape = ndarray.shape
    assert len(shape) == 1
    numel = shape[0]
    bp = open(name, 'rb')
    for i in range(numel):
        bytes = bp.read(4)
        vals = struct.unpack('f', bytes)
        val = vals[0]
        ndarray[i] = val
    return ndarray


def save_weights_as_miemienet(name, model_std, image_data_format, fuse_conv_bn, bn_eps, paddle_std=False):
    param_names = model_std.keys()
    content = ''
    bp = open('%s.bin' % name, 'wb')
    start_bytes_i = 0
    ele_bytes = 4
    param_num = 0
    for param_name in param_names:
        if '-' in param_name:
            print('\'-\' cant appear in param_name!')
            exit(0)
        if 'num_batches_tracked' in param_name:
            continue
        if 'anchor_points' in param_name:
            continue
        if 'stride_tensor' in param_name:
            continue
        if fuse_conv_bn:
            if '.conv.bias' in param_name:
                continue
            if '.norm.weight' in param_name:
                continue
            if '.norm.bias' in param_name:
                continue
            if '.norm.running_mean' in param_name:
                continue
            if '.norm.running_var' in param_name:
                continue
            if '.norm._mean' in param_name:
                continue
            if '.norm._variance' in param_name:
                continue
            if '.bn.weight' in param_name:
                continue
            if '.bn.bias' in param_name:
                continue
            if '.bn.running_mean' in param_name:
                continue
            if '.bn.running_var' in param_name:
                continue
            if '.bn._mean' in param_name:
                continue
            if '.bn._variance' in param_name:
                continue
            # 为了应付picodet中 DPModule 层的权重。
            if '.bn1.weight' in param_name:
                continue
            if '.bn1.bias' in param_name:
                continue
            if '.bn1._mean' in param_name:
                continue
            if '.bn1._variance' in param_name:
                continue
            if '.bn2.weight' in param_name:
                continue
            if '.bn2.bias' in param_name:
                continue
            if '.bn2._mean' in param_name:
                continue
            if '.bn2._variance' in param_name:
                continue

        weight = model_std[param_name]
        if isinstance(weight, dict):
            continue
        ndim = weight.ndim
        if paddle_std:
            ndarray = np.copy(weight)
        else:
            ndarray = weight.cpu().detach().numpy()
        conv_b_name = None
        new_conv_b = None
        # 简单判断一下，如果ndim==4，代表是卷积层的权重，如果ndim==2，代表是全连接层的权重
        if ndim == 4:
            if fuse_conv_bn:
                if '.conv.weight' in param_name:
                    conv_w = np.copy(ndarray)
                    target_bn_w_name1 = param_name.replace('.conv.weight', '.norm.weight')
                    target_bn_w_name2 = param_name.replace('.conv.weight', '.bn.weight')
                    if exist_key(param_names, target_bn_w_name1):
                        if paddle_std:
                            bn_w = model_std[param_name.replace('.conv.weight', '.norm.weight')]
                            bn_b = model_std[param_name.replace('.conv.weight', '.norm.bias')]
                            bn_m = model_std[param_name.replace('.conv.weight', '.norm._mean')]
                            bn_v = model_std[param_name.replace('.conv.weight', '.norm._variance')]
                        else:
                            bn_w = model_std[param_name.replace('.conv.weight', '.norm.weight')].cpu().detach().numpy()
                            bn_b = model_std[param_name.replace('.conv.weight', '.norm.bias')].cpu().detach().numpy()
                            bn_m = model_std[param_name.replace('.conv.weight', '.norm.running_mean')].cpu().detach().numpy()
                            bn_v = model_std[param_name.replace('.conv.weight', '.norm.running_var')].cpu().detach().numpy()
                    elif exist_key(param_names, target_bn_w_name2):
                        if paddle_std:
                            bn_w = model_std[param_name.replace('.conv.weight', '.bn.weight')]
                            bn_b = model_std[param_name.replace('.conv.weight', '.bn.bias')]
                            bn_m = model_std[param_name.replace('.conv.weight', '.bn._mean')]
                            bn_v = model_std[param_name.replace('.conv.weight', '.bn._variance')]
                        else:
                            bn_w = model_std[param_name.replace('.conv.weight', '.bn.weight')].cpu().detach().numpy()
                            bn_b = model_std[param_name.replace('.conv.weight', '.bn.bias')].cpu().detach().numpy()
                            bn_m = model_std[param_name.replace('.conv.weight', '.bn.running_mean')].cpu().detach().numpy()
                            bn_v = model_std[param_name.replace('.conv.weight', '.bn.running_var')].cpu().detach().numpy()
                    else:
                        raise NotImplementedError("nnnnnnnnn")
                    eps = bn_eps   # 要具体设置, 否则结果会稍有不同！！！
                    conv_b_name = param_name.replace('.conv.weight', '.conv.bias')
                    print(param_name)
                    if conv_b_name in param_names:
                        conv_b = model_std[conv_b_name].cpu().detach().numpy()
                    else:
                        conv_b = np.zeros(bn_w.shape)
                    new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
                    new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b
                    ndarray = new_conv_w
                # 为了应付picodet中 DPModule 层的权重。
                if '.dwconv.weight' in param_name:
                    conv_w = np.copy(ndarray)
                    bn_w = model_std[param_name.replace('.dwconv.weight', '.bn1.weight')]
                    bn_b = model_std[param_name.replace('.dwconv.weight', '.bn1.bias')]
                    bn_m = model_std[param_name.replace('.dwconv.weight', '.bn1._mean')]
                    bn_v = model_std[param_name.replace('.dwconv.weight', '.bn1._variance')]
                    eps = bn_eps   # 要具体设置, 否则结果会稍有不同！！！
                    conv_b_name = param_name.replace('.dwconv.weight', '.dwconv.bias')
                    print(param_name)
                    if conv_b_name in param_names:
                        conv_b = model_std[conv_b_name].cpu().detach().numpy()
                    else:
                        conv_b = np.zeros(bn_w.shape)
                    new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
                    new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b
                    ndarray = new_conv_w
                # 为了应付picodet中 DPModule 层的权重。
                if '.pwconv.weight' in param_name:
                    conv_w = np.copy(ndarray)
                    bn_w = model_std[param_name.replace('.pwconv.weight', '.bn2.weight')]
                    bn_b = model_std[param_name.replace('.pwconv.weight', '.bn2.bias')]
                    bn_m = model_std[param_name.replace('.pwconv.weight', '.bn2._mean')]
                    bn_v = model_std[param_name.replace('.pwconv.weight', '.bn2._variance')]
                    eps = bn_eps   # 要具体设置, 否则结果会稍有不同！！！
                    conv_b_name = param_name.replace('.pwconv.weight', '.pwconv.bias')
                    print(param_name)
                    if conv_b_name in param_names:
                        conv_b = model_std[conv_b_name].cpu().detach().numpy()
                    else:
                        conv_b = np.zeros(bn_w.shape)
                    new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
                    new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b
                    ndarray = new_conv_w
            if image_data_format == "NHWC":
                # [out_C, in_C, kH, kW] -> [kH, kW, in_C, out_C]
                ndarray = ndarray.transpose((2, 3, 1, 0))
        if ndim == 2:
            if image_data_format == "NHWC":
                # [out_C, in_C] -> [in_C, out_C]
                ndarray = ndarray.transpose((1, 0))
        if '.proj_conv.weight' in param_name or '.proj_conv.bias' in param_name:
            pass
            # cpname = copy.deepcopy(param_name)
            # nnn = 3
            # ndarray = np.reshape(ndarray, (-1, ))
            # numel = ndarray.shape[0]
            # print("aaaaaaaaaaaaaa")
            # print(ndarray)
            # for yrt in range(nnn):
            #     content += '%s,%d,%d,%d\n' % (cpname.replace('.proj_conv.', '.proj_convs.%d.' % yrt), start_bytes_i, ele_bytes, numel)
            #     param_num += 1
            #     start_bytes_i += ele_bytes * numel
            #     for i1 in range(numel):
            #         bp = bp_write_value(bp, ndarray[i1], force_fp32=True)
        else:
            ndarray = np.reshape(ndarray, (-1, ))
            numel = ndarray.shape[0]
            content += '%s,%d,%d,%d\n' % (param_name, start_bytes_i, ele_bytes, numel)
            param_num += 1
            start_bytes_i += ele_bytes * numel
            for i1 in range(numel):
                bp = bp_write_value(bp, ndarray[i1], force_fp32=True)
            if new_conv_b is not None:
                numel_b = new_conv_b.shape[0]
                content += '%s,%d,%d,%d\n' % (conv_b_name, start_bytes_i, ele_bytes, numel_b)
                param_num += 1
                start_bytes_i += ele_bytes * numel_b
                for i1 in range(numel_b):
                    bp = bp_write_value(bp, new_conv_b[i1], force_fp32=True)
    param_num_line = '%d\n' % (param_num, )
    content = param_num_line + content
    with open('%s.mie' % name, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()

