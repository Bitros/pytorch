import os
import pickle
from typing import Optional, Callable, Any, Tuple

import numpy as np
import torchvision
from PIL import Image
from torch.nn import Module, Linear, Conv2d, Sequential, MaxPool2d, Flatten, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset


class SeqExplorer(Module):
    def __init__(self):
        r"""
         Sequential module explorer CIFAR10 quick model.
        """
        super().__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, i):
        return self.seq(i)


class LocalCIFAR10(Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.data: Any = []
        self.targets = []
        self.train_list = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]

        self.test_list = [
            "test_batch",
        ]
        self.meta = {
            "filename": "batches.meta",
            "key": "label_names",
        }
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        path = os.path.join(self.root, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    train_ds = LocalCIFAR10(root='../input/cifar10-python/cifar-10-batches-py/', transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    seq = SeqExplorer()
    loss = CrossEntropyLoss()
    opt = SGD(seq.parameters(), lr=0.01)
    for epoch in range(20):
        running_loss = 0.0
        for images, targets in test_loader:
            outputs = seq(images)
            result_loss = loss(outputs, targets)
            opt.zero_grad()
            result_loss.backward()
            opt.step()
            running_loss = running_loss + result_loss
        print(running_loss)
