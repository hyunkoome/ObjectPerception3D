import os
import pickle
import typing
import numpy as np
import tensorflow as tf

from pathlib import Path
from utils import common_utils
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

# tf.enable_eager_execution() is a function in TensorFlow that switches the execution mode to 'eager execution'.
# Starting from TensorFlow 2.0, eager execution is enabled by default,
# so there's generally no need to call it separately.
# However, in TensorFlow 1.x versions, graph execution mode is the default, so tf.enable_eager_execution() is used
# to switch to eager execution mode before running code directly.
# try:
#     tf.enable_eager_execution()
# except:
#     pass


WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def extract_attributes(laser_labels: typing.List[typing.Any]) -> typing.Dict[str, np.array]:
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    num_points_in_gt, tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], [], []

    for label in laser_labels:
        box = label.box
        class_ind = label.type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(label.detection_difficulty_level)
        tracking_difficulty.append(label.tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])
        locations.append(loc)
        obj_ids.append(label.id)
        num_points_in_gt.append(label.num_lidar_points_in_box)
        speeds.append([label.metadata.speed_x, label.metadata.speed_y])
        accelerations.append([label.metadata.accel_x, label.metadata.accel_y])

    return {
        'name': np.array(obj_name),
        'difficulty': np.array(difficulty),
        'dimensions': np.array(dimensions),
        'location': np.array(locations),
        'heading_angles': np.array(heading_angles),
        'obj_ids': np.array(obj_ids),
        'tracking_difficulty': np.array(tracking_difficulty),
        'speed_global': np.array(speeds),
        'accel_global': np.array(accelerations)
    }


def pad_speed_global(annotations):
    """
    :param annotations: dictionary containing annotations
    :return: numpy array of padded speed_global annotations

    The `pad_speed_global` method takes in a `annotations` dictionary and pads the 'speed_global' annotation
        if it exists and has the correct shape.
    If 'speed_global' is not found or has an incorrect shape,
        it returns a numpy array of zeros with shape (annotations['name'].shape[0], 3).

    The 'speed_global' annotation is padded by adding an extra column with zeros to the existing 'speed_global' array.
    The padding is done using `np.pad` function with the `constant` mode and constant value of 0.

    Example usage:

    annotations = {
        'name': np.array(['A', 'B', 'C']),
        'speed_global': np.array([[1, 2], [3, 4], [5, 6]])
    }

    padded_speed_global = pad_speed_global(annotations)
    print(padded_speed_global)

    Output:
    [[1 2 0]
     [3 4 0]
     [5 6 0]]
    """
    if 'speed_global' in annotations and annotations['speed_global'].shape[1] == 2:
        return np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)
    else:
        return np.zeros(
            (annotations['name'].shape[0], 3))  # Default padding if 'speed_global' is not found or has incorrect shape


def compute_speed_in_pose(global_speed: np.array, pose: np.array) -> np.array:
    """
    :param global_speed: A numpy array representing the global speed.
    :param pose: A numpy array representing the pose.
    :return: A numpy array representing the computed speed.
    """
    speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
    return speed[:, :2]


def concatenate_annotations(annotations):
    """
    Concatenates annotations to create a single array of ground-truth boxes in LiDAR coordinates.

    :param annotations: a dictionary containing annotation information
    :return: an array of ground-truth boxes in LiDAR coordinates

    Example:
    annotations =
        'location': [...],
        'dimensions': [...],
        'heading_angles': [...],
        'speed_in_pose': [...]

    gt_boxes_lidar = concatenate_annotations(annotations)
    """
    gt_boxes_lidar = np.concatenate([
        annotations['location'],
        annotations['dimensions'],
        annotations['heading_angles'][..., np.newaxis],
        annotations['speed_in_pose']
    ], axis=1)
    return gt_boxes_lidar


def compute_gt_boxes_lidar(annotations, pose):
    """
    Compute the ground truth boxes in LiDAR coordinates.

    :param annotations: Dictionary containing the annotations.
    :param pose: Pose information for LiDAR calibration.
    :return: Ground truth boxes in LiDAR coordinates.
    """
    global_speed = pad_speed_global(annotations)
    speed_in_pose = compute_speed_in_pose(global_speed, pose)
    annotations['speed_in_pose'] = speed_in_pose
    gt_boxes_lidar = concatenate_annotations(annotations)
    return gt_boxes_lidar


def generate_labels(frame, pose):
    """
    Generate labels for the given frame and pose.

    :param frame: The frame object containing the laser labels.
    :param pose: The pose of the frame.
    :return: The generated annotations.
    """

    laser_labels = frame.laser_labels

    annotations = extract_attributes(laser_labels)
    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    annotations['gt_boxes_lidar'] = compute_gt_boxes_lidar(annotations, pose)

    return annotations


def extract_range_image_attributes(range_image, calib_info):
    """
    Extracts beam inclinations and extrinsic transform from range image calibration.

    Args:
        range_image: Waymo range image data.
        calib_info: Calibration information for the specific lidar.

    Returns:
        beam_inclinations: Beam inclinations for the lidar.
        extrinsic: Extrinsic transform matrix for the lidar.
    """
    # if len(calib_info.beam_inclinations) == 0:
    #     beam_inclinations = range_image_utils.compute_inclination(
    #         tf.constant([calib_info.beam_inclination_min, calib_info.beam_inclination_max]),
    #         height=range_image.shape.dims[0])
    # else:
    #     beam_inclinations = tf.constant(calib_info.beam_inclinations)

    if len(calib_info.beam_inclinations):
        beam_inclinations = tf.constant(calib_info.beam_inclinations)
    else:
        beam_inclinations = range_image_utils.compute_inclination(
            tf.constant([calib_info.beam_inclination_min, calib_info.beam_inclination_max]),
            height=range_image.shape.dims[0])

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(calib_info.extrinsic.transform), [4, 4])

    return beam_inclinations, extrinsic


def extract_range_image_cartesian(range_image_tensor, extrinsic, beam_inclinations, pixel_pose_local, frame_pose_local):
    """
    Extracts point cloud, NLZ, intensity, and elongation from range image tensor.

    Args:
        range_image_tensor: Tensor of range image data.
        extrinsic: Extrinsic transform matrix for the lidar.
        beam_inclinations: Beam inclinations for the lidar.
        pixel_pose_local: Pixel pose for the lidar range image.
        frame_pose_local: Frame pose for the lidar range image.

    Returns:
        a dictionary of
            tensors =
                dict of
                    points_tensor: Extracted 3D points from the range image.
                    points_NLZ_tensor: NLZ (Near, Low and Zero) values from the range image.
                    points_intensity_tensor: Intensity values from the range image.
                    points_elongation_tensor: Elongation values from the range image.
            range_image_mask = Mask indicating valid points in the range image.
    """
    # Create a mask where the first channel values are greater than 0
    # range_image_mask = range_image_tensor[..., 0] > 0
    range_image_mask = tf.math.greater(range_image_tensor[..., 0], 0)

    # range_image_nlz = range_image_tensor[..., 3]
    # range_image_intensity = range_image_tensor[..., 1]
    # range_image_elongation = range_image_tensor[..., 2]

    # Select values from indices 3, 1, and 2 of tensor
    range_image_nlz = tf.gather(range_image_tensor, indices=[3], axis=-1)
    range_image_intensity = tf.gather(range_image_tensor, indices=[1], axis=-1)
    range_image_elongation = tf.gather(range_image_tensor, indices=[2], axis=-1)

    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        range_image=tf.expand_dims(range_image_tensor[..., 0], axis=0),
        extrinsic=tf.expand_dims(extrinsic, axis=0),
        inclination=tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local
    )

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask))
    points_nlz_tensor = tf.gather_nd(range_image_nlz, tf.where(range_image_mask))
    points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.where(range_image_mask))
    points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.where(range_image_mask))

    return dict(tensors=dict(points_tensor=points_tensor.numpy(), points_nlz_tensor=points_nlz_tensor.numpy(),
                             points_intensity_tensor=points_intensity_tensor.numpy(),
                             points_elongation_tensor=points_elongation_tensor.numpy()),
                range_image_mask=range_image_mask)


def process_calibration(calib_info, range_images, camera_projections, range_image_top_pose_tensor, frame_pose,
                        ri_index):
    points_single, cp_points_single = [], []
    points_nlz_single, points_intensity_single, points_elongation_single = [], [], []
    for current_ri_index in ri_index:
        range_image = range_images[calib_info.name][current_ri_index]
        beam_inclinations, extrinsic = extract_range_image_attributes(range_image=range_image,
                                                                      calib_info=calib_info)
        range_image_tensor = tf.reshape(tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if calib_info.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)

        results_dict = extract_range_image_cartesian(range_image_tensor=range_image_tensor, extrinsic=extrinsic,
                                                     beam_inclinations=beam_inclinations,
                                                     pixel_pose_local=pixel_pose_local,
                                                     frame_pose_local=frame_pose_local)

        cp = camera_projections[calib_info.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(results_dict['range_image_mask']))
        points_single.append(results_dict['tensors']['points_tensor'])
        cp_points_single.append(cp_points_tensor)
        points_nlz_single.append(results_dict['tensors']['points_nlz_tensor'])
        points_intensity_single.append(results_dict['tensors']['points_intensity_tensor'])
        points_elongation_single.append(results_dict['tensors']['points_elongation_tensor'])

    return dict(points_single=points_single, cp_points_single=cp_points_single, points_nlz_single=points_nlz_single,
                points_intensity_single=points_intensity_single, points_elongation_single=points_elongation_single)


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose,
                                       ri_index=(0, 1)):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)

    # Convert the frame pose to a tensor.
    frame_pose_tensor = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # Convert and reshape the top pose to a tensor.
    range_image_top_pose_tensor = tf.convert_to_tensor(range_image_top_pose.data)
    range_image_top_pose_tensor = tf.reshape(range_image_top_pose_tensor, range_image_top_pose.shape.dims)
    # Get rotation and translation matrices.
    rotation_matrix = transform_utils.get_rotation_matrix(range_image_top_pose_tensor[..., 0],
                                                          range_image_top_pose_tensor[..., 1],
                                                          range_image_top_pose_tensor[..., 2])
    translation_matrix = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(rotation_matrix, translation_matrix)

    points, cp_points, points_nlz, points_intensity, points_elongation = [], [], [], [], []

    for calib_info in calibrations:
        calib_results = process_calibration(calib_info=calib_info, range_images=range_images,
                                            camera_projections=camera_projections,
                                            range_image_top_pose_tensor=range_image_top_pose_tensor,
                                            frame_pose=frame_pose_tensor,
                                            ri_index=ri_index)

        points.append(np.concatenate(calib_results['points_single'], axis=0))
        cp_points.append(np.concatenate(calib_results['cp_points_single'], axis=0))
        points_nlz.append(np.concatenate(calib_results['points_nlz_single'], axis=0))
        points_intensity.append(np.concatenate(calib_results['points_intensity_single'], axis=0))
        points_elongation.append(np.concatenate(calib_results['points_elongation_single'], axis=0))

    return points, cp_points, points_nlz, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, use_two_returns=True):
    """
    Save Lidar Points

    Save Lidar points from a given frame to a specified path.

    :param frame: The input frame.
    :param cur_save_path: The path where the Lidar points will be saved.
    :param use_two_returns: Optional. If True, uses two returns from the Lidar. Default is True.
    :return: A list containing the number of points from each Lidar.

    """
    # ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    # if len(ret_outputs) == 4:
    #     range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    # else:
    #     assert len(ret_outputs) == 3
    #     range_images, camera_projections, range_image_top_pose = ret_outputs

    frame_results = frame_utils.parse_range_image_and_camera_projection(frame=frame)
    range_images, camera_projections, seg_labels, range_image_top_pose = frame_results

    points, cp_points, points_in_nlz_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame=frame, range_images=range_images, camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_nlz_flag = np.concatenate(points_in_nlz_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([points_all, points_intensity,
                                  points_elongation, points_in_nlz_flag], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar


def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True,
                            update_info_only=False):
    # sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    sequence_name = Path(sequence_file).stem

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = Path(save_path).joinpath(sequence_name)
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    # pkl_file = Path(cur_save_dir).joinpath('%s.pkl'.format(sequence_name))
    pkl_file = Path(cur_save_dir).joinpath('{}.pkl'.format(sequence_name))

    sequence_infos = []
    sequence_infos_old = None

    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        # sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos
