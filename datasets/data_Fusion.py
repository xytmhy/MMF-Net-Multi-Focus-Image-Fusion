import os.path
import glob
import fusion_transforms
from datasets.listdataset import ListDataset
from .util import split2list

'''
Using our artificat fusion dataset
Thanks to the marvelous work of FlowNet.
'''

def make_dataset(dataset_dir, split):
    img1_dir = 'InputL'
    assert(os.path.isdir(os.path.join(dataset_dir, img1_dir)))
    img2_dir = 'InputR'
    assert (os.path.isdir(os.path.join(dataset_dir, img2_dir)))
    gt_dir = 'FusionGT'
    assert(os.path.isdir(os.path.join(dataset_dir, gt_dir)))
    map_dir = 'FusionMapT'
    assert (os.path.isdir(os.path.join(dataset_dir, map_dir)))

    images = []

    for map in sorted(glob.glob(os.path.join(dataset_dir, map_dir, '*.png'))):
        map = os.path.relpath(map, os.path.join(dataset_dir, map_dir))

        scene_dir, filename = os.path.split(map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')

        img1 = os.path.join(img1_dir, scene_dir, '{}_L.png'.format(prefix))
        img2 = os.path.join(img2_dir, scene_dir, '{}_R.png'.format(prefix))
        gt = os.path.join(gt_dir, scene_dir, '{}_FR.png'.format(prefix))
        map = os.path.join(map_dir, map)
        if not (os.path.isfile(os.path.join(dataset_dir, img1)) or os.path.isfile(os.path.join(dataset_dir, map))):
            continue
        images.append([[img1, img2, gt], map])
        # put the input and the deblur result together, since they are RGB images and the deblur estimation is single channel

    return split2list(images, split, default_split=0.9)

def fusion_data(root, transform=None, target_transform=None, co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform, None)

    return train_dataset, test_dataset
