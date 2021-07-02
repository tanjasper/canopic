# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from PIL import Image
from data_fns import get_paths


class FDDBDataset(object):

    def __init__(
        self, root, filenames_path, transforms=None
    ):
        super(FDDBDataset, self).__init__()

        self.orig_root = 'datasets/fddb/images/orig'  # for getting original image dimensions

        self.filenames_path = filenames_path
        self.paths = get_paths(filenames_path, root)
        self.orig_paths = get_paths(filenames_path, self.orig_root)
        self.categories = {'face': 1}
        self._transforms = transforms
        self.num_im = len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        # obtain the image paths
        im_path = self.paths[idx % self.num_im]
        # load images (grayscale for direct inference)
        img = Image.open(im_path).convert('RGB')
        # perform transformations
        if self._transforms is not None:
            img, target = self._transforms(img, None)
        return img, None, idx

    def get_img_info(self, index):
        # TODO: save this info somewhere and just load it here instead of loading each image
        im_path = self.orig_paths[index % self.num_im]
        img = Image.open(im_path).convert('RGB')
        return {"height": img.size[1], "width": img.size[0]}
