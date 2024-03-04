import torch.cuda

from lib.models.regression.model import RegressionModel
from lib.models.matching.model import FeatureMatchingModel


def build_model(cfg, checkpoint='', use_loftr_preds=False, use_superglue_preds=False, args=None):
    if cfg.MODEL == 'FeatureMatching':
        return FeatureMatchingModel(cfg)
    elif cfg.MODEL == 'Regression':
        model = RegressionModel(cfg, use_loftr_preds=use_loftr_preds, use_superglue_preds=use_superglue_preds, 
                                ckpt_path=checkpoint, inference=True, use_vanilla_transformer=args.use_vanilla_transformer,
                                d_model=args.d_model, max_steps=args.max_steps, use_prior=args.use_prior) \
                                    if \
            checkpoint != '' else RegressionModel(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    else:
        raise NotImplementedError()





