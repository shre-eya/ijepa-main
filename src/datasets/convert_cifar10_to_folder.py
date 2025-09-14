import os
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
import random

def convert(split):
    dataset = CIFAR10(root="./data", train=(split=="train"), download=False)
    base_dir = f"./data/cifar10/{split}"
    os.makedirs(base_dir, exist_ok=True)

    if split == "train":
        # Group indices by class
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        # For each class, randomly select 4000 indices
        selected_indices = []
        for indices in class_indices.values():
            selected_indices.extend(random.sample(indices, 4000))
    else:
        # For test, use all images in the test split
        selected_indices = list(range(len(dataset)))

    for new_idx, idx in enumerate(tqdm(selected_indices, desc=f"Converting {split} set")):
        img, label = dataset[idx]
        class_name = dataset.classes[label]
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{new_idx}.png")
        save_image(transforms.ToTensor()(img), img_path)

if __name__ == "__main__":
    convert("train")
    convert("test")  # If needed, rename it to 'val' manually
