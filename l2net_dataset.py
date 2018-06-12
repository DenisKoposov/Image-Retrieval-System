import os
import errno
import random
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.utils.data as data

import pickle as pkl
from utils import download_url, check_integrity


class PhotoTour(data.Dataset):
    urls = {
        'notredame_harris': [
            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
            'notredame_harris.zip',
            '69f8c90f78e171349abdf0307afefe4d'
        ],
        'yosemite_harris': [
            'http://matthewalunbrown.com/patchdata/yosemite_harris.zip',
            'yosemite_harris.zip',
            'a73253d1c6fbd3ba2613c45065c00d46'
        ],
        'liberty_harris': [
            'http://matthewalunbrown.com/patchdata/liberty_harris.zip',
            'liberty_harris.zip',
            'c731fcfb3abb4091110d0ae8c7ba182c'
        ],
        'notredame': [
            'http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip',
            'notredame.zip',
            '509eda8535847b8c0a90bbb210c83484'
        ],
        'yosemite': [
            'http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip',
            'yosemite.zip',
            '533b2e8eb7ede31be40abc317b2fd4f0'
        ],
        'liberty': [
            'http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip',
            'liberty.zip',
            'fdd9152f138ea5ef2091746689176414'
        ],
    }
    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437,
            'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019,
           'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}
    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092,
            'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}
    image_ext = 'bmp'
    info_file = 'info.txt'
    interest_file = 'interest.txt'

    def __init__(self, root, name, train=True,
                 p1=64, p2=64, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_down = os.path.join(self.root, '{}.zip'.format(name))
        self.images_file = os.path.join(self.root, '{}_img.pt'.format(name))
        self.idx_file = os.path.join(self.root, '{}_idx.pkl'.format(name))

        self.train = train
        self.transform = transform
        self.mean = self.mean[name]
        self.std = self.std[name]
        self.p1 = p1
        self.p2 = p2
        self._pos = 0

        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.images = torch.load(self.images_file)
        with open(self.idx_file, 'rb') as f:
            self.idx = pkl.load(f)
        #print("Data type:", type(self.images), type(self.idx))

    def __getitem__(self, index):
        """
            Do not use batch_size option of DataLoader.
            Use p1 and p2 instead.
        """
        data1 = [ random.sample(patch_idx, 2) for patch_idx in self.idx[self._pos:self._pos+self.p1] ]
        random_points = random.sample(self.idx[:self._pos] + self.idx[self._pos+self.p1:], self.p2)
        data2 = [ random.sample(patch_idx, 2) for patch_idx in random_points ]
        self._pos += self.p1
        data1 = data1 + data2
        data1 = list(zip(*data1))
        data2 = self.images[list(data1[1])]
        data1 = self.images[list(data1[0])]

        #print(data1.shape, data2.shape)

        if self.transform is not None:
            data1 = torch.cat([self.transform(patch).unsqueeze(0)
                               for patch in data1], 0)
            data2 = torch.cat([self.transform(patch).unsqueeze(0)
                               for patch in data2], 0)
        
        #print(data1.shape, data2.shape)

        return data1, data2

    def __len__(self):
        return int(len(self.idx) / self.p1)

    def _check_datafile_exists(self):
        return os.path.exists(self.images_file) and os.path.exists(self.idx_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}, {}'.format(self.images_file,
                                                      self.idx_file))
            return

        if not self._check_downloaded():
            # download files
            url = self.urls[self.name][0]
            filename = self.urls[self.name][1]
            md5 = self.urls[self.name][2]
            fpath = os.path.join(self.root, filename)

            download_url(url, self.root, filename, md5)

            print('# Extracting data {}\n'.format(self.data_down))

            import zipfile
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.data_dir)

            os.unlink(fpath)

        # process and save as torch files
        print('# Caching data')

        images = read_image_file(self.data_dir, self.image_ext, self.lens[self.name])
        points = read_info_file(self.data_dir, self.info_file)
        #refImg = read_interest_file(self.data_dir, self.interest_file)

        print('# Formatting data')
        #print(images.shape, len(points))
        idx = []
        i = 0
        last = len(images)
        min_len = 100

        while i < last:
            point = points[i]
            #print(i, last, point, points[i])
            one_point = []
            while i < last and points[i] == point:
                one_point.append(i)
                i += 1
            #print(len(one_point))
            if min_len > len(one_point):
                min_len = len(one_point)
            idx.append(one_point)

        print("minimal number of patches:", min_len)
        print("Saving to file")

        with open(self.images_file, 'wb') as f:
            torch.save(images, f)
        #print("Idx length:", len(idx))
        with open(self.idx_file, 'wb') as f:
            pkl.dump(idx, f)
        print("Saved")

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def read_image_file(data_dir, image_ext, n):
    """Return a Tensor containing the patches
    """
    def PIL2array(_img):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir, _image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(PIL2array(patch))
    return torch.ByteTensor(np.array(patches[:n]))


def read_info_file(data_dir, info_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, info_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)


def read_interest_file(data_dir, interest_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, interest_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)