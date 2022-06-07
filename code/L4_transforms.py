from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import L1_custom_dataset


def tensor_exp():
    writer.add_image('tensor_exp', transforms.ToTensor()(img))


def crop_exp():
    writer.add_image('crop_exp', transforms.CenterCrop(64)(img_tensor))


def normalize_exp():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    writer.add_image('normalize_exp', normalize(img_tensor))


def resize_exp():
    writer.add_image('resize_exp', transforms.Resize((512, 512))(img_tensor))


def compose_exp():
    compose_trans = transforms.Compose((transforms.CenterCrop(256), transforms.Resize(500), transforms.ToTensor()))
    writer.add_image('compose_exp', compose_trans(img))


def random_crop_exp():
    for i in range(10):
        writer.add_image('random_crop_exp', transforms.RandomCrop(100)(img_tensor), i)

if __name__ == '__main__':
    # 1286984635_5119e80de1.jpg
    # 500 * 375 HWC
    img = my_dataset.ants_dataset[8][0]
    img_tensor = transforms.ToTensor()(img)
    with SummaryWriter('../logs') as writer:
        # tensor_exp()
        # crop_exp()
        # normalize_exp()
        # resize_exp()
        # compose_exp()
        random_crop_exp()
