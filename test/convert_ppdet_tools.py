import os
import pickle
import six
from test2_utils import save_as_txt, save_weights_as_txt, save_weights_as_miemienet, load_ckpt
import argparse


def make_parser():
    parser = argparse.ArgumentParser("convert ppdet tools.")
    parser.add_argument(
        "--model_path", default="", help="model_path"
    )
    return parser




'''
cd test

wget https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams

wget https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams


python convert_ppdet_tools.py --model_path ppyoloe_crn_s_300e_coco.pdparams

python convert_ppdet_tools.py --model_path picodet_s_416_coco_lcnet.pdparams


'''


if __name__ == "__main__":
    args = make_parser().parse_args()
    model_path = args.model_path

    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
    backbone_dic = {}
    fpn_dic = {}
    head_dic = {}
    others = {}
    for key, value in state_dict.items():
        if 'tracked' in key:
            continue
        if 'backbone' in key:
            backbone_dic[key] = value
        elif 'neck' in key:
            fpn_dic[key] = value
        elif 'head' in key:
            head_dic[key] = value
        else:
            others[key] = value


    save_name = model_path.split('.')[0]
    print('save_name=%s'%save_name)
    miemienet_image_data_format = "NHWC"
    fuse_conv_bn = True
    paddle_std = True

    os.makedirs('save_data', exist_ok=True)

    print('save ...')
    save_weights_as_miemienet("save_data/%s" % (save_name, ), state_dict, miemienet_image_data_format, fuse_conv_bn, 1e-5, paddle_std)
    print('saved!')





