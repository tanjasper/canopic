import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import torch.utils.data as data
from torchvision.datasets.folder import make_dataset
import numpy as np
import math
import re
import sys
import torch


# Currently unused
class DualClassDataset:

    # data_dirs and filename_locs must be list of strings
    def __init__(self, data_dirs, filenames_locs, transform):
        self.data_dirs = data_dirs
        self.filename_locs = filenames_locs
        self.transform = transform

        # obtain paths and labels
        self.paths = []
        self.labels = []
        self.sublabels = []
        for idx, filename in enumerate(self.filename_locs):
            curr_paths = get_paths(filename, self.data_dirs[idx])
            self.paths = self.paths + curr_paths
            self.labels = self.labels + len(self.paths) * [idx]
        self.num_im = len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # obtain the image paths
        im_path = self.paths[index % self.num_im]
        label = self.labels[index % self.num_im]
        sublabel = self.sublabels[index % self.num_im]
        # load images (grayscale for direct inference)
        im = Image.open(im_path).convert('RGB')
        # perform transformations
        if self.transform is not None:
            im = self.transform(im)
        return im, label, sublabel


# Class for loading data from a single dataset with a single filenames text
# Assigns each subdirectory as a label.
# Input:
#   filenames_loc -- path to txt file where each line contains the relative path to the image
#   data_dir -- a string that is appended to every line on filenames_loc to get absolute paths to the images
#   transform -- a Pytorch torchvision transform that is applied to every image
class DatasetFromFilenames:

    def __init__(self, data_dir, filenames_loc, transform):
        self.data_dir = data_dir
        self.filenames = filenames_loc
        self.transform = transform
        self.paths = get_paths(self.filenames, self.data_dir)
        self.labels, self.class_counts = labels_from_consecutive_subdir(self.paths)
        self.num_im = len(self.paths)
        self.weights = 1. / np.array(self.class_counts)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # obtain the image paths
        im_path = self.paths[index % self.num_im]
        im_label = self.labels[index % self.num_im]
        # load images (grayscale for direct inference)
        im = Image.open(im_path).convert('RGB')
        # perform transformations
        if self.transform is not None:
            im = self.transform(im)
        return im, im_label


# Class for loading data from multiple filenames
# Each filenames will constitute one class/label (i.e. giving two filenames txt files will give two classes)
# Input:
#   filenames_loc -- a list of filenames txt files, each similar to DatasetFromFilenames
#   data_dir -- a list of the strings appended to the corresponding filenames txt file
#   transform -- a Pytorch torchvision transform that is applied to every image
class DatasetFromMultipleFilenames:

    # data_dirs and filename_locs must be list of strings
    def __init__(self, data_dirs, filenames_locs, transform):
        self.data_dirs = data_dirs
        self.filename_locs = filenames_locs
        self.transform = transform

        # obtain paths and labels
        self.paths = []
        self.labels = []
        for idx, filename in enumerate(self.filename_locs):
            curr_paths = get_paths(filename, self.data_dirs[idx])
            self.paths = self.paths + curr_paths
            self.labels = self.labels + len(self.paths)*[idx]
        self.num_im = len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # obtain the image paths
        im_path = self.paths[index % self.num_im]
        label = self.labels[index % self.num_im]
        # load images (grayscale for direct inference)
        im = Image.open(im_path).convert('RGB')
        # perform transformations
        if self.transform is not None:
            im = self.transform(im)
        return im, label


class Pytorch1RandomSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

# function for getting image paths out from filenames text file
def get_paths(fname, dir=''):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            temp = str(line).strip()
            paths.append(os.path.join(dir, temp))
    return paths


# assigns labels to a list of filenames according to their directories
# that is, all images in a directory share one unique label
def labels_from_subdir(paths):
    labels = []
    subdirs = []
    for i in range(len(paths)):
        curr_subdir = os.path.dirname(paths[i])
        if curr_subdir not in subdirs:
            subdirs.append(curr_subdir)
        labels.append(subdirs.index(curr_subdir))
    return labels


def labels_from_consecutive_subdir(paths):
    labels = []
    class_counts = []
    curr_idx = 0
    curr_class_count = 0
    for i in range(len(paths)):
        curr_subdir = os.path.dirname(paths[i])
        if i == 0:
            prev_subdir = curr_subdir
        if prev_subdir != curr_subdir:
            curr_idx += 1
            prev_subdir = curr_subdir
            class_counts.append(curr_class_count)
            curr_class_count = 1
        else:
            curr_class_count += 1
        labels.append(curr_idx)
    class_counts.append(curr_class_count)
    return labels, class_counts

# DatasetFolder that takes in multiple roots. Each root pertains to one class
# Modified from torchvision.datasets.DatasetFolder
class MultipleDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, roots, loader, extensions, transform=None, target_transform=None):

        samples = []
        root_lengths = []

        for i in range(len(roots)):
            root = roots[i]
            classes, class_to_idx = self._find_classes(root)
            temp_samples = make_dataset(root, class_to_idx, extensions)  # should be the path names of the images
            samples = samples + temp_samples
            root_lengths.append(len(samples))
            if len(samples) == 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                   "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        #self.targets = [s[1] for s in samples]
        self.targets = [i for (i, s) in enumerate(root_lengths) for a in range(s)]
        # Above line makes a list of idxs for each image where the idx is the root idx
        # s is the root_length, i is the root_idx. For each s, repeat idx i s times.

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = self.targets[index]  # override DatasetFolder's target
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MultipleImageFolder(MultipleDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader):
        super(MultipleImageFolder, self).__init__(roots, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


# copy-pasted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder or dataset_type is MultipleImageFolder:  # i changed this
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples