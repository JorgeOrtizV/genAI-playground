from torch.utils.data import Dataset
from PIL import Image
import os

class customDataset(Dataset):
    def __init__(self, root_dir, transform=None, channels=1):
        self.root_dir = root_dir
        self.channels = channels
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.channels ==1:
            img = Image.open(img_path).convert("L")
        else:
            img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label since your model expects a (data, label) tuple