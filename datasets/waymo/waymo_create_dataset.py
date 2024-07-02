import argparse
import yaml
import os
import pickle
import multiprocessing
from pathlib import Path

from datasets.waymo.create_waymo_dataset import CreateWaymoDataset
from utils.create_dataset_utils import create_logger


def create_waymo_infos(dataset_cfg, class_names, data_and_save_path):
    workers = min(16, multiprocessing.cpu_count())
    dataset = CreateWaymoDataset(dataset_cfg=dataset_cfg,
                                 class_names=class_names,
                                 root_path=data_and_save_path,
                                 training=False,
                                 logger=create_logger())

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split('train')
    waymo_infos_train = dataset.get_infos(raw_data_path=Path.joinpath(data_and_save_path, 'raw_data'),
                                          save_path=Path.joinpath(data_and_save_path,
                                                                  dataset_cfg['PROCESSED_DATA_TAG']),
                                          num_workers=workers,
                                          has_label=True,
                                          sampled_interval=1)

    train_filename = Path.joinpath(data_and_save_path,
                                   '{}_infos_{}.pkl'.format(dataset_cfg['PROCESSED_DATA_TAG'], 'train'))
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)

    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split('test')
    waymo_infos_val = dataset.get_infos(raw_data_path=Path.joinpath(data_and_save_path, 'raw_data'),
                                        save_path=Path.joinpath(data_and_save_path, dataset_cfg['PROCESSED_DATA_TAG']),
                                        num_workers=workers,
                                        has_label=True,
                                        sampled_interval=1)

    val_filename = Path.joinpath(data_and_save_path,
                                 '{}_infos_{}.pkl'.format(dataset_cfg['PROCESSED_DATA_TAG'], 'test'))
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)

    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split('train')
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=data_and_save_path, split='train', sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=dataset_cfg['PROCESSED_DATA_TAG']
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    args = parser.parse_args()

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))

    dataset_cfg = dict(yaml_config)
    dataset_cfg['PROCESSED_DATA_TAG'] = args.processed_data_tag

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    create_waymo_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        data_and_save_path=Path(os.path.join(ROOT_DIR, 'data', 'waymo'))
    )
