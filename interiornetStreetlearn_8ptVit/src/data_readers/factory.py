
# RGBD-Dataset
from .streetlearn import StreetLearn
from .interiornet import InteriorNet

def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = { 'streetlearn': (StreetLearn, ), 
                    'interiornet': (InteriorNet, ),
    }
    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)
    return ConcatDataset(db_list)
