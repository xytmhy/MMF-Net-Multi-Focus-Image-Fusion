import argparse
import os
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import fusion_transforms
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot

os.environ['CUDA_VISIBLE_DEVICES']='0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch CBAfusionNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default='/media/data2/mhy/LytroDataset',
                    help='path to images folder, image names must match \'[name]A.[ext]\' and \'[name]B.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', default='/home/mhy/PycharmProjects/CBAfusion-master/fusion_data/10-27-08:17/CBAfusionnet,adam,1000epochs,b36,lr0.0001/model_best.pth.tar',
                    help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'ResultsNew'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading code
    input_transform = transforms.Compose([
        fusion_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*A.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + 'B.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.cat([img1, img2]).unsqueeze(0)

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)

        for fusion_output in output:
            results = tensor2rgb(fusion_output)
            result_fusion = results[1:, :]
            result_Gmap = results[:1, :]

            to_save = np.rint(result_fusion * 255)
            to_save[to_save<0] = 0
            to_save[to_save>255] = 255
            to_save = to_save.astype(np.uint8).transpose(1, 2, 0)
            pyplot.imsave(save_path / '{}{}.png'.format(img1_file.namebase[:-1], 'Fusion'), to_save)

            to_save = np.rint(result_Gmap * 255)
            to_save[to_save < 0] = 0
            to_save[to_save > 255] = 255
            to_save = to_save.astype(np.uint8).transpose(1, 2, 0)
            to_save = np.squeeze(to_save)
            pyplot.imsave(save_path / '{}{}.png'.format(img1_file.namebase[:-1], 'Guidance'), to_save)


def tensor2rgb(img_tensor):

    map_np = img_tensor.detach().cpu().numpy()

    return map_np


if __name__ == '__main__':
    main()
