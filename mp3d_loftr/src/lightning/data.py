import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)

from src.datasets.mp3d import Mp3dDataset, Mp3dLightDataset
from src.datasets.interiornet_streetlearn import InteriornetStreetlearnDataset

class InteriornetStreetlearnDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()


        self.args = args
        self.config = config

        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
        self.val_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }
        self.test_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }

    def setup(self, stage=None):
        assert stage in ['fit', 'test']

        if stage == 'fit':
            self.train_dataset = InteriornetStreetlearnDataset(self.args.numpy_data_path, self.args.dset_name, 'train', 
                                            from_saved_preds=self.args.from_saved_preds,
                                            full_train_set=self.args.save_preds is not None)
            self.val_dataset = InteriornetStreetlearnDataset(self.args.numpy_data_path, self.args.dset_name, 'val', 
                                            from_saved_preds=self.args.from_saved_preds,
                                            full_train_set=self.args.save_preds is not None)
        else:
            self.test_dataset = InteriornetStreetlearnDataset(self.args.numpy_data_path, self.args.dset_name, 'test', split=self.args.eval_split,
                                            from_saved_preds=self.args.from_saved_preds,
                                            full_train_set=self.args.save_preds is not None)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)

class Mp3dDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config

        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
        self.val_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }
        self.test_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }

    def setup(self, stage=None):
        assert stage in ['fit', 'test']

        if stage == 'fit':
            self.train_dataset = Mp3dDataset(self.config.DATASET.TRAIN_DATA_JSON, self.config.DEPTH_DIR, self.config.DATA_DIR, 'train', 
                                            from_saved_preds=self.args.from_saved_preds,
                                            from_saved_corr=self.args.from_saved_corr,
                                            full_train_set=self.args.save_preds is not None,
                                            use_large_dset=self.args.use_large_dset,
                                            use_40pct_dset=self.args.use_40pct_dset,
                                            load_prior_ransac=self.args.load_prior_ransac,)
            self.val_dataset = Mp3dDataset(self.config.DATASET.VAL_DATA_JSON, self.config.DEPTH_DIR, self.config.DATA_DIR, 'val', 
                                            from_saved_preds=self.args.from_saved_preds,
                                            from_saved_corr=self.args.from_saved_corr,
                                            full_train_set=self.args.save_preds is not None,
                                            use_large_dset=self.args.use_large_dset,
                                            use_40pct_dset=self.args.use_40pct_dset,
                                            load_prior_ransac=self.args.load_prior_ransac,)
        else:
            all_jsons = {"train": self.config.DATASET.TRAIN_DATA_JSON, 
                         "val": self.config.DATASET.VAL_DATA_JSON, 
                         "test": self.config.DATASET.TEST_DATA_JSON}
            self.test_dataset = Mp3dDataset(all_jsons[self.args.eval_split], self.config.DEPTH_DIR, self.config.DATA_DIR, 'test', split=self.args.eval_split,
                                            load_predictions_path=self.args.load_predictions_path,
                                            from_saved_preds=self.args.from_saved_preds,
                                            from_saved_corr=self.args.from_saved_corr,
                                            full_train_set=self.args.save_preds is not None,
                                            load_prior_ransac=self.args.load_prior_ransac,)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)

class Mp3dLightDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config

        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
        self.val_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }
        self.test_loader_params = {
            'batch_size': 1,
            'num_workers': 2
        }

    def setup(self, stage=None):
        assert stage in ['fit', 'test']

        if stage == 'fit':
            self.train_dataset = Mp3dLightDataset(self.config.DATASET.TRAIN_DATA_JSON, 'train',
                                                 correspondences_use_fit_only=self.args.correspondences_use_fit_only,
                                                 correspondence_transformer_load_feats=self.args.correspondence_tf_use_feats,
                                                 max_correspondences=self.args.max_correspondences,
                                                 use_pred_corr=self.args.use_pred_corr,
                                                from_saved_preds=self.args.from_saved_preds,
                                                 outlier_pct=self.args.outlier_pct,
                                                 noise_pix=self.args.noise_pix,
                                                 missing_pct=self.args.missing_pct,
                                                 after_ransac_path=self.args.after_ransac_path,
                                                corr_dropout=self.args.corr_dropout,
                                                use_large_dset=self.args.use_large_dset,
                                                no_use_loftr_preds=self.args.solver == "prior_ransac" or self.args.solver == "prior_ransac_noprior")
            self.val_dataset = Mp3dLightDataset(self.config.DATASET.VAL_DATA_JSON, 'val',
                                                 correspondences_use_fit_only=self.args.correspondences_use_fit_only,
                                                 correspondence_transformer_load_feats=self.args.correspondence_tf_use_feats,
                                                 max_correspondences=self.args.max_correspondences,
                                                from_saved_preds=self.args.from_saved_preds,
                                                 use_pred_corr=self.args.use_pred_corr,
                                                 outlier_pct=self.args.outlier_pct,
                                                 noise_pix=self.args.noise_pix,
                                                 missing_pct=self.args.missing_pct,
                                                 after_ransac_path=self.args.after_ransac_path,
                                                corr_dropout=self.args.corr_dropout,
                                                use_large_dset=self.args.use_large_dset,
                                                no_use_loftr_preds=self.args.solver == "prior_ransac" or self.args.solver == "prior_ransac_noprior")
        else:
            all_jsons = {"train": self.config.DATASET.TRAIN_DATA_JSON, 
                         "val": self.config.DATASET.VAL_DATA_JSON, 
                         "test": self.config.DATASET.TEST_DATA_JSON}
            self.test_dataset = Mp3dLightDataset(all_jsons[self.args.eval_split], 'test', split=self.args.eval_split,
                                                 correspondences_use_fit_only=self.args.correspondences_use_fit_only,
                                                 correspondence_transformer_load_feats=self.args.correspondence_tf_use_feats,
                                                 max_correspondences=self.args.max_correspondences,
                                                from_saved_preds=self.args.from_saved_preds,
                                                 use_pred_corr=self.args.use_pred_corr,
                                                 outlier_pct=self.args.outlier_pct,
                                                 noise_pix=self.args.noise_pix,
                                                 missing_pct=self.args.missing_pct,
                                                 after_ransac_path=self.args.after_ransac_path,
                                                corr_dropout=self.args.corr_dropout,
                                                use_large_dset=self.args.use_large_dset,
                                                no_use_loftr_preds=self.args.solver == "prior_ransac" or self.args.solver == "prior_ransac_noprior")

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)

