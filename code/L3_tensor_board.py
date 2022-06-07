from torch.utils.tensorboard import SummaryWriter
import numpy as np
import L1_custom_dataset

with SummaryWriter('../logs') as writer:
    for i in range(100):
        writer.add_scalar('y=x', i, i)

with SummaryWriter('../logs') as writer:
    for idx, (img, _) in enumerate(L1_custom_dataset.ants_dataset):
        writer.add_image('img_array', np.array(img), idx, dataformats='HWC')
