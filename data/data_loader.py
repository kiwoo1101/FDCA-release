import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, poseid = self.dataset[index]
        spl = img_path.split('@f')
        if len(spl) == 1:
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(spl[0]).convert('RGB')
            img = T.RandomHorizontalFlip(p=1)(img)

        if self.transform is not None:
            img = self.transform(img)

        # return img, pid, camid, img_path
        return img, pid, camid, poseid


if __name__=='__main__':
    s = 'agd.jpg'
    str = s.split('@f')
    print(len(str))
    pass