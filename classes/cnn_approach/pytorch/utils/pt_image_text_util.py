import torch
import os
import tempfile
import PIL

from PIL import ImageOps
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional
from pathlib import Path


class PTImageTextDataset(Dataset):
    def __init__(self, images: Optional[List[str]], tokens: Optional[List[int]],
                 image_root_path: Optional[str], image_shape: Optional[Tuple],
                 temp_img_path: Optional[str] = None,
                 labels: Optional[List[List[int]]] = None):

        self.images = images
        self.tokens = tokens
        self.labels = labels

        self.image_root_path = image_root_path
        self.image_shape = image_shape

        self.temp_img_path = temp_img_path
        if self.temp_img_path is None:
            self.temp_img_path = tempfile.mkdtemp()
        else:
            os.makedirs(self.temp_img_path, exist_ok=True)

    def __len__(self):
        if self.images is not None:
            return len(self.images)
        else:
            return len(self.tokens)

    def __getitem__(self, idx):
        image = None
        text = None
        label = None

        if self.images is not None:
            image_path = self.images[idx]
            cache_path = str(Path(f"{self.temp_img_path}{image_path}.jpg").resolve())

            if os.path.exists(cache_path):
                image = read_image(cache_path)
            else:
                image = PIL.Image.open(image_path)

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image.thumbnail(self.image_size, PIL.Image.LANCZOS)
                image = ImageOps.pad(image, size=self.image_size, color=0, centering=(0.5, 0.5))

                image.save(cache_path)

                image = TF.to_tensor(image)

            image = image.float()

        if self.tokens is not None:
            text = torch.Tensor(self.tokens[idx])

        if self.labels is not None:
            label = self.labels[idx]

        #         print(f"Get item for idx {idx} - Image: {image}, Text: {text}, Label: {label}")

        result = [image, text, label]
        result = list(filter(lambda a: a is not None, result))

        return tuple(result)
