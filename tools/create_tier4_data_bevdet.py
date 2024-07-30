import pickle
import numpy as np
from nuscenes import NuScenes
from tools.data_converter import tier4dataset_converter as tier4_converter
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    "animal": 'ignore',
    "movable_object.barrier": 'barrier',
    "movable_object.debris": 'ignore',
    "movable_object.pushable_pullable": 'ignore',
    "movable_object.trafficcone": 'traffic_cone',
    "pedestrian.adult": 'pedestrian',
    "pedestrian.child": 'pedestrian',
    "pedestrian.construction_worker": 'pedestrian',
    "pedestrian.personal_mobility": 'ignore',
    "pedestrian.police_officer": 'pedestrian',
    "pedestrian.stroller": 'ignore',
    "pedestrian.wheelchair": 'ignore',
    "vehicle.car": 'car',
    "vehicle.construction": 'construction_vehicle',
    "vehicle.emergency (ambulance & police)": 'ignore',
    "vehicle.motorcycle": 'motorcycle',
    "vehicle.trailer": 'trailer',
    "vehicle.truck": 'truck',
    "vehicle.bicycle": 'bicycle',
    "vehicle.bus (bendy & rigid)": 'bus',
    "static_object.bicycle_rack": 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels

def tier4_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to TIER4 dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    tier4_converter.create_tier4_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def add_ann_adj_info(dataroot, extra_tag):
    sample_nums = {}
    for set in ['train', 'val']:#只处理train和val
        dataset = pickle.load(
            open(dataroot + '/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        sample_nums[set] = len(dataset['infos'])
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            scene_path = dataroot + '/' + info['scene_dir']#当前关键帧对应的场景文件夹
            nuscenes = NuScenes(version='annotation', dataroot=scene_path)
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
        with open(dataroot + '/%s_infos_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)
    return sample_nums

if __name__ == '__main__':
    dataset = 'tier4'
    version = 'Odaiba_JT_v1.0'
    root_path = './data/Odaiba_JT_v1.0'
    extra_tag = 'bevdetv2-tier4'
    tier4_data_prep(
        root_path= root_path,
        info_prefix = extra_tag,
        version= version,
        max_sweeps = 0)
    print('add_ann_infos')
    sample_nums = add_ann_adj_info(root_path, extra_tag)
    #统计sample的数量
    print("all samples num:", sample_nums)
    