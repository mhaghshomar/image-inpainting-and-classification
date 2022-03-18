import pytorch_lightning as pl
import typing as th
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from .utils import InpaintingData, initialize_transforms
from .. import utils as osail_utils


@DATAMODULE_REGISTRY
class InpaintingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            # dataset
            root: th.Union[str, Dataset],
            val_root: th.Optional[str] = None,
            train_dataset_params: th.Optional[dict] = None,
            val_dataset_params: th.Optional[dict] = None,
            dataset_params: th.Optional[dict] = None,
            val_size: th.Optional[th.Union[int, float]] = None,

            # batch_size
            batch_size: th.Optional[int] = 16,
            train_batch_size: th.Optional[int] = None,
            val_batch_size: th.Optional[int] = None,
            # shuffle
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            # num_workers
            num_workers: int = 0,
            train_num_workers: th.Optional[int] = None,
            val_num_workers: th.Optional[int] = None,
            # seed
            seed: int = 0,
            # extra parameters (ignored)
            **kwargs
    ):
        super().__init__(train_transforms=None, val_transforms=None)
        # seed
        self.seed = seed

        # data
        self.root = root
        self.val_root = val_root
        self.train_data, self.val_data = None, None

        self.val_size = val_size
        assert val_size is None or isinstance(val_size, int) or (isinstance(val_size, float) and val_size < 1), \
            "invalid validation size is provided (either int, float (between zero and one) or None)"

        self.train_dataset_params = {**(dataset_params or dict()), **(train_dataset_params or dict())}
        self.val_dataset_params = {**(dataset_params or dict()), **(val_dataset_params or dict())}

        # batch_size
        self.train_batch_size = train_batch_size if train_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        assert self.train_batch_size or self.val_batch_size or self.test_batch_size, \
            'at least one of batch_sizes should be a positive number'

        # shuffle
        self.train_shuffle, self.val_shuffle = train_shuffle, val_shuffle

        # num_workers
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers


    def setup(self, stage: th.Optional[str] = None) -> None:
        if stage == 'fit' and self.train_batch_size:
            dataset = InpaintingData(self.root, **self.train_dataset_params)
            if not self.val_size or self.val_root:
                self.train_data = dataset
            if self.val_size and self.val_batch_size and not self.val_root:
                train_len = len(dataset) - self.val_size if isinstance(self.val_size, int) else int(
                    len(dataset) * (1 - self.val_size))
                prng = torch.Generator()
                prng.manual_seed(self.seed)
                train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len], generator=prng)
                self.train_data, self.val_data = train_data, val_data
            if self.val_root:
                self.train_data = dataset
                self.val_data = InpaintingData(self.val_root, **self.val_dataset_params) if self.val_batch_size else None
        
    def get_dataloader(self, name, batch_size=None, shuffle=None, num_workers=None, **params):
        data = getattr(self, f'{name}_data')
        if data is None:
            return None
        batch_size = batch_size if batch_size is not None else getattr(self, f'{name}_batch_size')
        shuffle = shuffle if shuffle is not None else getattr(self, f'{name}_shuffle')
        num_workers = num_workers if num_workers is not None else getattr(self, f'{name}_num_workers')
        return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **params)

    def train_dataloader(self, **params):
        return self.get_dataloader('train', **params)

    def val_dataloader(self, **params):
        return self.get_dataloader('val', **params)



@DATAMODULE_REGISTRY
class ClassificationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            # dataset
            root: th.Union[str, Dataset],
            val_k: int = None,
            dataset_params: th.Optional[dict] = None,

            # batch_size
            batch_size: th.Optional[int] = 16,
            train_batch_size: th.Optional[int] = None,
            val_batch_size: th.Optional[int] = None,
            # shuffle
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            # num_workers
            num_workers: int = 0,
            train_num_workers: th.Optional[int] = None,
            val_num_workers: th.Optional[int] = None,
            # seed
            seed: int = 0,
            # extra parameters (ignored)
            **kwargs
    ):
        super().__init__(train_transforms=None, val_transforms=None)
        # seed
        self.seed = seed

        # data
        self.root = root
        self.val_k = val_k
        self.train_data, self.val_data = None, None

        assert val_k is None or isinstance(val_k, int), \
            "invalid validation k is provided (expected int value)"

        self.dataset_params = dataset_params or dict()

        # batch_size
        self.train_batch_size = train_batch_size if train_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        assert self.train_batch_size or self.val_batch_size or self.test_batch_size, \
            'at least one of batch_sizes should be a positive number'

        # shuffle
        self.train_shuffle, self.val_shuffle = train_shuffle, val_shuffle

        # num_workers
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers


    def setup(self, stage: th.Optional[str] = None) -> None:
        if stage == 'fit' and self.train_batch_size:
            self.train_data = ConcatDataset([ImageFolder(f'{self.root}/{i}') for i in range(5) if self.val_k is None or i != self.val_k])
            self.val_data = ImageFolder(f'{self.root}/{self.val_k}') if self.val_k is not None else None
            
    def get_dataloader(self, name, batch_size=None, shuffle=None, num_workers=None, **params):
        data = getattr(self, f'{name}_data')
        if data is None:
            return None
        batch_size = batch_size if batch_size is not None else getattr(self, f'{name}_batch_size')
        shuffle = shuffle if shuffle is not None else getattr(self, f'{name}_shuffle')
        num_workers = num_workers if num_workers is not None else getattr(self, f'{name}_num_workers')
        return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **params)

    def train_dataloader(self, **params):
        return self.get_dataloader('train', **params)

    def val_dataloader(self, **params):
        return self.get_dataloader('val', **params)
