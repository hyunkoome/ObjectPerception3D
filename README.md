
conda activate centerpoint

python setup.py develop

python -m datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset_use_feature_no_elongation.yaml 


python datasets/waymo/waymo_create_dataset.py --cfg_file /home/hyunkoo/DATA/LiDAR/ObjectPerception3D/tools/cfgs/dataset_configs/waymo_dataset_use_feature_no_elongation.yaml

python datasets/waymo/waymo_create_dataset.py --cfg_file /home/hyunkoo/DATA/LiDAR/ObjectPerception3D/tools/cfgs/dataset_configs/waymo_dataset_ref.yaml
