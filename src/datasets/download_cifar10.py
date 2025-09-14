from torchvision.datasets import CIFAR10
from torchvision import transforms

CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
