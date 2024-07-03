import argparse
import yaml
import os
import pickle
import multiprocessing
import typing
from pathlib import Path

from datasets.waymo.create_waymo_dataset import CreateWaymoDataset
from utils.create_dataset_utils import create_logger


def _construct_path(path: Path, folder: str) -> Path:
    """Helper function to construct a Path object."""
    return Path.joinpath(path, folder)


def _format_filename(data_tag: str, split_name: str) -> str:
    """Helper function to format the filename string."""
    return f"{data_tag}_infos_{split_name}.pkl"


def create_dataset_info_file(dataset: CreateWaymoDataset,
                             data_and_save_path: typing.Union[str, Path],
                             cfg_dataset: typing.Dict[str, typing.Any],
                             split_name: str,
                             workers: int) -> typing.Union[str, Path]:
    """
    Create an info file for a given dataset instance, data path and dataset configuration.

    :param dataset: Instance of the CreateWaymoDataset class.
    :param data_and_save_path: Path to the dataset and save directory.
    :param cfg_dataset: Dictionary containing dataset configuration.
    :param split_name: Name of the data split (e.g., 'train', 'test').
    :param workers: Number of workers for parallel processing.
    :return: Filename of the saved info file.
    """
    dataset.set_split(split_name)
    raw_data_path = _construct_path(data_and_save_path, 'raw_data')
    save_path = _construct_path(data_and_save_path, cfg_dataset['PROCESSED_DATA_TAG'])
    waymo_infos = dataset.get_dataset_infos(raw_data_path=raw_data_path,
                                            save_path=save_path,
                                            num_workers=workers,
                                            has_label=True,
                                            sampled_interval=1
                                            )
    filename = _construct_path(data_and_save_path, _format_filename(cfg_dataset['PROCESSED_DATA_TAG'], split_name))
    with open(filename, 'wb') as f:
        pickle.dump(waymo_infos, f)
    print(f'-----------Waymo info file for {split_name} is saved to {filename}---------------')
    return filename


def waymo_dataset_preprocess(cfg_dataset: typing.Dict, class_names: typing.List[str],
                             data_and_save_path: typing.Union[Path, str]) -> None:
    """
    This method performs data preprocessing for the Waymo dataset.
    It creates a dataset object using the provided dataset configuration, class names, and data path.
    It then generates data
    * information files for the 'train' and 'test' splits of the dataset.
    * After that, it creates a groundtruth database for data augmentation using a specified set of classes.
    * Finally, it prints a message indicating that the data preparation is done.

    :param cfg_dataset: A dictionary containing the configuration parameters for the dataset.
    :param class_names: A list of strings representing the names of the classes in the dataset.
    :param data_and_save_path: The path to the data and the location where the processed data will be saved.
    :return: None
    """
    workers = min(16, multiprocessing.cpu_count())
    dataset = CreateWaymoDataset(dataset_cfg=cfg_dataset,
                                 class_names=class_names,
                                 root_path=data_and_save_path,
                                 training=False,
                                 logger=create_logger())
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    train_filename = create_dataset_info_file(dataset=dataset, data_and_save_path=data_and_save_path,
                                              cfg_dataset=cfg_dataset,
                                              split_name='train', workers=workers)
    _ = create_dataset_info_file(dataset=dataset, data_and_save_path=data_and_save_path, cfg_dataset=cfg_dataset,
                                 split_name='test', workers=workers)

    print('---------------Start create groundtruth database for data augmentation---------------')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset.set_split('train')
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=data_and_save_path, split='train', sampled_interval=1,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=cfg_dataset['PROCESSED_DATA_TAG']
    )
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


if __name__ == '__main__':
    args = parse_arguments()
    dataset_config = load_and_prepare_config(args)
    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()

    CLASS_NAMES_CONST = ['Vehicle', 'Pedestrian', 'Cyclist']

    waymo_dataset_preprocess(
        cfg_dataset=dataset_config,
        class_names=CLASS_NAMES_CONST,
        data_and_save_path=Path(os.path.join(ROOT_DIR, 'data', 'waymo'))
    )
