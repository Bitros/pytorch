from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataSet(Dataset):

    def __init__(self, root, label_dir):
        self.root_dir = root
        self.label_dir = label_dir
        self.path = os.path.join(root, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):
        return self.img_path.__len__()


root_dir = '../data/train/'
ants_label_dir = 'ants_image'
bees_label_dir = 'bees_image'
ants_dataset = MyDataSet(root_dir, ants_label_dir)
bees_dataset = MyDataSet(root_dir, ants_label_dir)
train_dataset = ants_dataset + bees_dataset
