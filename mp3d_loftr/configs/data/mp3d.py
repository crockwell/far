from configs.data.base import cfg

DATA_DIR = 'data/mp3d_rpnet_v4_sep20'

cfg.DATASET.TRAINVAL_DATA_SOURCE = "mp3d"
cfg.DATASET.TEST_DATA_SOURCE = "mp3d"

cfg.DATASET.TRAIN_DATA_JSON = f"{DATA_DIR}/mp3d_planercnn_json/cached_set_train.json"
cfg.DATASET.VAL_DATA_JSON = f"{DATA_DIR}/mp3d_planercnn_json/cached_set_val.json"
cfg.DATASET.TEST_DATA_JSON = f"{DATA_DIR}/mp3d_planercnn_json/cached_set_test.json"

cfg.DEPTH_DIR = f"{DATA_DIR}/observations"
cfg.DATA_DIR = f"{DATA_DIR}"