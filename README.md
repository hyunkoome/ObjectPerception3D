
conda activate centerpoint

python setup.py develop

python -m datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset_use_feature_no_elongation.yaml 