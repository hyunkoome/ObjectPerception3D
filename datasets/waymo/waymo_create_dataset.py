import argparse
import yaml
import os
import pickle
import multiprocessing
import typing
from pathlib import Path

from datasets.waymo.create_waymo_dataset import CreateWaymoDataset
from utils.create_dataset_utils import create_logger


def waymo_dataset_preprocess(cfg_dataset: typing.Dict) -> None:
    """
    This method performs data preprocessing for the Waymo dataset.
    It creates a dataset object using the provided dataset configuration, class names, and data path.
    It then generates data
    * information files for the 'train' and 'test' splits of the dataset.
    * After that, it creates a groundtruth database for data augmentation using a specified set of classes.
    * Finally, it prints a message indicating that the data preparation is done.

    :param cfg_dataset: A dictionary containing the configuration parameters for the dataset.
    :return: None
    """
    # The path to the data and the location where the processed data will be saved.

    root_dir = get_root_dir()

    workers = min(16, multiprocessing.cpu_count())
    dataset = CreateWaymoDataset(dataset_cfg=cfg_dataset,
                                 root_path=root_dir,
                                 logger=create_logger())

    print('---------------Start to generate data infos---------------')
    # To disable all CUDA devices, not to use GPUs, all computations use only CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print('train')
    dataset.set_split(split_name='train')
    train_filename = dataset.get_dataset_infos(num_workers=workers, has_label=True, split_name='train')

    print('test')
    dataset.set_split(split_name='test')
    _ = dataset.get_dataset_infos(num_workers=workers, has_label=True, split_name='test')

    print('---------------Start create groundtruth database for data augmentation---------------')
    # Activate GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split('train')
    dataset.create_groundtruth_database(info_file_path=train_filename, split_mode='train')
    print('---------------Data preparation Done---------------')


def parse_arguments() -> argparse.Namespace:
    """
    This method `parse_arguments` is used to parse command line arguments.

    :return: An instance of `argparse.Namespace` containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--config_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    return parser.parse_args()


def load_and_prepare_config(args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
    """
    Load and prepare configuration.

    :param args: The command line arguments.
    :type args: argparse.Namespace
    :return: The prepared configuration dataset.
    :rtype: typing.Dict[str, typing.Any]
    """
    yaml_config = yaml.safe_load(open(args.config_file))
    config_dataset = dict(yaml_config)
    config_dataset['PROCESSED_DATA_TAG'] = args.data_tag
    return config_dataset


def get_root_dir() -> Path:
    """Obtain the root directory of the project."""
    return Path(__file__).parent.parent.parent.resolve()


if __name__ == '__main__':
    args = parse_arguments()
    dataset_config = load_and_prepare_config(args)
    waymo_dataset_preprocess(cfg_dataset=dataset_config['DATASET'])
