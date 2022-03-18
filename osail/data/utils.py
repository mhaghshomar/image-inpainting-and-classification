import torch
from torchvision import transforms as tv_transforms
import os
import torchvision as tv
import PIL
import typing as th
from .. import utils as osail_utils

class InpaintingData(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        extension="png",
        max_box_size: int = 100,
        min_box_size: int = 50,
        **kwargs,
    ):
        self.root = root
        self.files = []
        self.min_box_size, self.max_box_size = min_box_size, max_box_size
        for dir_name, sub_dirs, files in os.walk(self.root):
            self.files += [
                f"{dir_name}/{file}" for file in files if file.endswith(extension)
            ]
        self.input_shape = self[0][0].shape

    def __getitem__(self, idx):
        img = tv.transforms.functional.to_tensor(PIL.Image.open(self.files[idx]))
        w, h = img.shape[1:]
        start = torch.randint(low=0, high=max(w, h), size=(2,))
        end = start + torch.randint(
            low=self.min_box_size, high=self.max_box_size, size=(2,)
        )
        obscured = img.clone()
        obscured[:, start[0] : end[0], start[1] : end[1]] = 0
        return obscured, img

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f'root: "{self.root}", len: {len(self)}'


def initialize_transforms(transforms: th.Optional[th.Union[list, dict, th.Any]]):
    if transforms is None or not isinstance(transforms, (dict, list)):
        return transforms

    if isinstance(transforms, list):
        return tv_transforms.Compose([initialize_transforms(i) for i in transforms])

    return osail_utils.get_value(transforms['cls'])(**transforms.get('args', dict()))
