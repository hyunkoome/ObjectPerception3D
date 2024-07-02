import os
import pickle
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from functools import partial

from ops.roiaware_pool3d import roiaware_pool3d_utils
from datasets.dataset import DatasetTemplate
from utils.create_dataset_utils import check_sequence_name_with_all_version


class CreateWaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = os.path.join(self.root_path, self.dataset_cfg['PROCESSED_DATA_TAG'])
        self.split = dict(train=self.dataset_cfg['DATA_SPLIT']['train'], test=self.dataset_cfg['DATA_SPLIT']['test'])

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )

        split_dir = os.path.join(self.root_path, 'ImageSets', '{}.txt'.format(self.split[split]))
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.seq_name_to_infos = self.include_waymo_data(mode=split)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = os.path.join(self.data_path, sequence_name, '{}.pkl'.format(sequence_name))
            info_path = check_sequence_name_with_all_version(info_path)

            if not os.path.exists(info_path):
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg['SAMPLED_INTERVAL'][mode] > 1:
            sampled_waymo_infos = []
            # for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
            for k in range(0, len(self.infos), self.dataset_cfg['SAMPLED_INTERVAL'][mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG',
                                                 None) is not None and self.dataset_cfg['SEQUENCE_CONFIG']['ENABLED']
        if not use_sequence_data:
            seq_name_to_infos = None
        return seq_name_to_infos

    # OK
    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True,
                  sampled_interval=1):
        from datasets.waymo import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            check_sequence_name_with_all_version(Path(os.path.join(raw_data_path, sequence_file)))
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = os.path.join(self.data_path, sequence_name, '{:04d}.npy'.format(sample_idx))
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        if self.dataset_cfg.get('POINTS_TANH_DIM', None) is None:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        else:
            for dim_idx in self.dataset_cfg['POINTS_TANH_DIM']:
                points_all[:, dim_idx] = np.tanh(points_all[:, dim_idx])
        return points_all

    # OK
    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                    processed_data_tag=None):
        database_save_path = save_path / (
                '%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / (
                '%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / (
                '%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in tqdm(range(0, len(infos), sampled_interval)):
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float32)
                    assert gt_points.dtype == np.float32
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)
