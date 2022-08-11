import os
import tempfile
import PIL

from PIL import ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path


class PTImageTextDataset(Dataset):
    """
    This is a subclass of torch.utils.data.Dataset, returning image and tokenised input based on the input args.
    images (List[str]): a list of image file names which the files are saved in image_root_path. To create a dataset
                        without images, set it to None
    tokens: ([Union[List[int], Dict]]): a list of token ids or a dictionary of tokens ids and additional fields
                                        required for transformer embedding. To create a dataset without tokens,
                                        set it to None
    image_root_path (Optional[str]): The path that stores the original images. This is optional if this
                                     is a text only dataset.
    image_shape (Optional[Tuple]): Image shape to be converted, the input should have three values -
                                   width, height and number of channels. This is optional if this
                                   is a text only dataset.
    transformed_img_path (Optional[str]): The path that stores the transformed images. This is optional if this
                                          is a text only dataset.
    labels (Optional[List[int]]): The labels to be predicted. This is optional if the data contains no labels.

    """
    def __init__(self,
                 images: Optional[List[str]],
                 tokens: Optional[Union[List[int], Dict]],
                 image_root_path: Optional[str] = None,
                 image_shape: Optional[Tuple] = None,
                 transformed_img_path: Optional[str] = None,
                 labels: Optional[List[int]] = None):

        self.images = images
        self.tokens = tokens
        self.labels = labels

        if images is not None:
            if image_root_path is None or image_shape is None:
                raise ValueError("A dataset with images should have image_root_path and image_shape")

            self.image_root_path = image_root_path
            self.image_shape = image_shape

            self.transformed_img_path = transformed_img_path
            if self.transformed_img_path is None:
                self.transformed_img_path = tempfile.mkdtemp()
            else:
                os.makedirs(self.transformed_img_path, exist_ok=True)

    def __len__(self) -> int:
        """
        This overrides __len__ method of the torch.utils.data.Dataset, returning the length of the dataset
        Returns:
            int:
                length of the dataset
        """
        if self.images is not None:
            return len(self.images)
        else:
            if isinstance(self.tokens, dict):
                for key, value in self.tokens.items():
                    return len(value)

            return len(self.tokens)

    def __getitem__(self, idx) -> Tuple:
        """
        This overrides __getitem__ method of the torch.utils.data.Dataset, returning the data the dataset at
        particular index
        Args:
            idx:
                index of the dataset

        Returns:
            Tuple:
                Data of the dataset at the given index. The returned format depends on the input args. The standard
                one is (images, text, labels). If any one of images, text and labels is missing from input, it will be
                eliminated from the returned value. (i.e. image only dataset - (images, labels), text only dataset -
                (text, labels))

        """

        image = None
        text = None
        label = None

        if self.images is not None:
            image_path = self.images[idx]
            cache_path = str(Path(f"{self.transformed_img_path}{image_path}.jpg").resolve())

            if os.path.exists(cache_path):
                image = read_image(cache_path)
            else:
                image = PIL.Image.open(image_path)

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image.thumbnail(self.image_shape, PIL.Image.LANCZOS)
                image = ImageOps.pad(image, size=self.image_shape, color=0, centering=(0.5, 0.5))

                image.save(cache_path)

                image = TF.to_tensor(image)

            image = image.float()

        if self.tokens is not None:
            if isinstance(self.tokens, dict):
                text = {key: value[idx] for key, value in self.tokens.items()}
            else:
                text = self.tokens[idx]

        if self.labels is not None:
            label = self.labels[idx]

        result = [image, text, label]
        result = list(filter(lambda a: a is not None, result))

        return tuple(result)

    @classmethod
    def get_dataloader_from_dataset(cls, ds: Dataset, batch_size: int) -> DataLoader:
        """
        This generates a torch.utils.data.DataLoader from the dataset and batch_size
        Args:
            ds: The dataset to be used in dataloader
            batch_size: The number of data to be loaded by the DataLoader in a single batch

        Returns:
            DataLoader:
                The dataloader of the given dataset.

        """
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
