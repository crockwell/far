import torch.utils as utils
from torchvision.transforms import ColorJitter, Grayscale
import pytorch_lightning as pl
from torch.distributed import get_rank, get_world_size, is_initialized

from lib.datasets.sampler import RandomConcatSampler
from lib.datasets.mapfree import MapFreeDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, use_loftr_preds=False, use_superglue_preds=False):
        super().__init__()
        self.cfg = cfg

        datasets = {'MapFree': MapFreeDataset}

        assert cfg.DATASET.DATA_SOURCE in datasets.keys(), 'invalid DATA_SOURCE, this dataset is not implemented'
        self.dataset_type = datasets[cfg.DATASET.DATA_SOURCE]
        self.use_loftr_preds = use_loftr_preds
        self.use_superglue_preds = use_superglue_preds

    def get_sampler(self, dataset, reset_epoch=False):
        if self.cfg.TRAINING.SAMPLER == 'scene_balance':
            sampler = RandomConcatSampler(dataset,
                                          self.cfg.TRAINING.N_SAMPLES_SCENE,
                                          self.cfg.TRAINING.SAMPLE_WITH_REPLACEMENT,
                                          shuffle=True,
                                          reset_on_iter=reset_epoch,
                                          rank=self.rank,
                                          num_replicas=self.num_replicas
                                          )
        else:
            sampler = None
        return sampler

    def train_dataloader(self):
        if is_initialized():
            self.rank = get_rank()
            self.num_replicas = get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

        transforms = ColorJitter() if self.cfg.DATASET.AUGMENTATION_TYPE == 'colorjitter' else None
        transforms = Grayscale(
            num_output_channels=3) if self.cfg.DATASET.BLACK_WHITE else transforms

        dataset = self.dataset_type(self.cfg, 'train', transforms=transforms, use_loftr_preds=self.use_loftr_preds,
                                    use_superglue_preds=self.use_superglue_preds)
        sampler = self.get_sampler(dataset)
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.TRAINING.BATCH_SIZE,
                                           num_workers=self.cfg.TRAINING.NUM_WORKERS,
                                           sampler=sampler
                                           )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'val', use_loftr_preds=self.use_loftr_preds, use_superglue_preds=self.use_superglue_preds)
        sampler = None
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.TRAINING.BATCH_SIZE,
                                           num_workers=self.cfg.TRAINING.NUM_WORKERS,
                                           sampler=sampler,
                                           drop_last=True
                                           )
        return dataloader

    def test_dataloader(self):
        dataset = self.dataset_type(self.cfg, 'test', use_loftr_preds=self.use_loftr_preds, use_superglue_preds=self.use_superglue_preds)
        dataloader = utils.data.DataLoader(dataset,
                                           batch_size=1,
                                           num_workers=1,
                                           shuffle=False)
        return dataloader
