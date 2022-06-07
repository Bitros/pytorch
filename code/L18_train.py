import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
try:
    # TPU
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(device)
except ImportError:
    pass


class CIFAR10QuickModel(Module):
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


if __name__ == '__main__':
    train_data = CIFAR10('../data', train=True, download=True, transform=ToTensor())
    test_data = CIFAR10('../data', train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    model = CIFAR10QuickModel().to(device)
    loss_fn = CrossEntropyLoss().to(device)
    lr = 1e-2
    optimizer = SGD(model.parameters(), lr=lr)
    epoch = 10
    total_train_step = 0
    total_test_step = 0
    for idx in range(epoch):
        print(f'Epoch {idx}')

        # set train mode
        model.train()
        for data in train_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f'Train step {total_train_step} loss {loss.item():.4f}')

        # set eval mode
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                total_test_step += 1
                if total_train_step % 100 == 0:
                    accuracy = (outputs.argmax(dim=1) == targets).sum().item()
                    print(f'accuracy {accuracy:.4f}')
        print(f'Test loss {total_test_loss:.4f}')
    torch.save(model.state_dict(), './model.pth')
