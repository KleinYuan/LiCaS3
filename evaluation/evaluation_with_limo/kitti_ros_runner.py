# This script deserialize data from a ROS bag, running inference and save the data in a separate rosbags

import fire
import rosbag
import cv2
import ros_numpy
import sensor_msgs
from collections import deque, Counter
import numpy as np
from copy import deepcopy
from utils import pb_server
from utils.kitti_loader import load_calibrations
from utils import projection
from box import Box
import yaml
import random

"""
path:        /xxxx/04.bag.missync.bag
version:     2.0
duration:    28.1s
start:       Dec 31 1969 16:00:00.00 (0.00)
end:         Dec 31 1969 16:00:28.11 (28.11)
size:        2.1 GB
messages:    3524
compression: none [1085/1085 chunks]
types:       geometry_msgs/PoseStamped [d3812c3cbc69362b77dc0b19b345f8f5]
             sensor_msgs/CameraInfo    [c9a58c1b0b154e0e6da7578cb991d214]
             sensor_msgs/Image         [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/PointCloud2   [1158d486dd51d683ce2f1be655c3c181]
             tf2_msgs/TFMessage        [94810edda583a504dfda3829e70d7eec]
topics:      /groundtruth_pose/pose                         271 msgs    : geometry_msgs/PoseStamped
             /sensor/camera/color/left/camera_info          271 msgs    : sensor_msgs/CameraInfo   
             /sensor/camera/color/left/image_rect           271 msgs    : sensor_msgs/Image        
             /sensor/camera/color/right/camera_info         271 msgs    : sensor_msgs/CameraInfo   
             /sensor/camera/color/right/image_rect          271 msgs    : sensor_msgs/Image        
             /sensor/camera/color_labels/left/camera_info   271 msgs    : sensor_msgs/CameraInfo   
             /sensor/camera/color_labels/left/image_rect    271 msgs    : sensor_msgs/Image        
             /sensor/camera/grayscale/left/camera_info      271 msgs    : sensor_msgs/CameraInfo   
             /sensor/camera/grayscale/left/image_rect       271 msgs    : sensor_msgs/Image        
             /sensor/camera/grayscale/right/camera_info     271 msgs    : sensor_msgs/CameraInfo   
             /sensor/camera/grayscale/right/image_rect      271 msgs    : sensor_msgs/Image        
             /sensor/velodyne/cloud_euclidean               271 msgs    : sensor_msgs/PointCloud2  
             /tf                                            271 msgs    : tf2_msgs/TFMessage       
             /tf_static                                       1 msg     : tf2_msgs/TFMessage
"""

"""
Iteration order
[833252400] Topic: /groundtruth_pose/pose
[833252400] Topic: /tf
[833252400] Topic: /sensor/velodyne/cloud_euclidean
[833252400] Topic: /sensor/camera/grayscale/left/image_rect
[833252400] Topic: /sensor/camera/grayscale/left/camera_info
[833252400] Topic: /sensor/camera/grayscale/right/image_rect
[833252400] Topic: /sensor/camera/grayscale/right/camera_info
[833252400] Topic: /sensor/camera/color_labels/left/image_rect
[833252400] Topic: /sensor/camera/color_labels/left/camera_info
[833252400] Topic: /sensor/camera/color/left/image_rect
[833252400] Topic: /sensor/camera/color/left/camera_info
[833252400] Topic: /sensor/camera/color/right/image_rect
[833252400] Topic: /sensor/camera/color/right/camera_info
[937505700] Topic: /groundtruth_pose/pose
[937505700] Topic: /tf
[937505700] Topic: /sensor/velodyne/cloud_euclidean
[937505700] Topic: /sensor/camera/grayscale/left/image_rect
[937505700] Topic: /sensor/camera/grayscale/left/camera_info
[937505700] Topic: /sensor/camera/grayscale/right/image_rect
[937505700] Topic: /sensor/camera/grayscale/right/camera_info
[937505700] Topic: /sensor/camera/color_labels/left/image_rect
[937505700] Topic: /sensor/camera/color_labels/left/camera_info
[937505700] Topic: /sensor/camera/color/left/image_rect
[937505700] Topic: /sensor/camera/color/left/camera_info
[937505700] Topic: /sensor/camera/color/right/image_rect
[937505700] Topic: /sensor/camera/color/right/camera_info

"""


def process_a_bag_with_tf_protobuf(bag_fp, benchmark_config_template_fp, models_config_fp, dataset_name, model_type,
                                   model_id):
    benchmark_config_template = Box(yaml.load(open(benchmark_config_template_fp, 'r').read())).benchmarks
    models_config = Box(yaml.load(open(models_config_fp, 'r').read()))

    model_count = 0
    models_config.models[dataset_name] = {model_id: models_config.models[dataset_name][model_id]}
    print("Only output the demo for model : {}".format(model_id))
    print("models_config.models[dataset_name]: {}".format(models_config.models[dataset_name]))
    test_model_info = models_config.models[dataset_name][model_id]
    updated_config = deepcopy(benchmark_config_template)
    s = int(test_model_info.s)
    l = int(test_model_info.l)
    model_fp = test_model_info.model_fp
    print("[{}  / {}] Testing model id {}, s = {}, l = {}\n   model path = {}".format(
        model_count, len(models_config.models[dataset_name]), model_id, s, l, model_fp))

    # TODO(kaiwen): I am assuming you always have a timestamp appended.
    #  Otherwise, you will need to change this line.
    model_namespace = model_fp.split('/')[-2].split('@')[0]

    updated_config.pb_fp = model_fp
    # TODO: the tensor section is generated on the fly instead
    if model_type == 'g':
        raise Exception("g model is not supported yet!")
    else:
        updated_config.tensors = deepcopy(updated_config.h_tensors)
        updated_config.tensors.inputs[0] = \
            updated_config.h_tensors.inputs[0].format(model_namespace)
        updated_config.tensors.outputs[0] = \
            updated_config.h_tensors.outputs[0].format(model_namespace)

    updated_config.training_data.sampling_window = l
    updated_config.training_data.sampling_stride = s
    updated_config.training_data.features.X.C = (l + 1) * 4

    assert (l == 5) or (
                l == 10), "It shall be (l == 5) or (l == 10), not supported. If you need to, write your own test."
    assert s == 1, "s != 1, not supported. If you need to, write your own test."

    print("Step3: Initialize evaluator ...")
    evaluator = pb_server.PBServer(updated_config)

    missync_bag = rosbag.Bag(bag_fp)
    licas3_bag = rosbag.Bag(f"{bag_fp}.licas3_bag.{model_id}.bag", 'w')
    previous_grayscale_frames = deque(maxlen=l * 2)
    previous_color_frames = deque(maxlen=l * 2)
    previous_lidar_frames = deque(maxlen=l * 2)
    offset_cnt = 0
    # Assumption:  the topics are in the order as mentioned in the top of the script
    # Assumption:  l = 5, s = 1
    camera_sensor_H, camera_sensor_W = updated_config.sensors_info.camera.shape
    data_info = load_calibrations(updated_config.raw_data.root_dir)
    T = data_info['T']
    R = data_info['R']
    P = data_info['P']
    crop_h_start = int(updated_config.training_data.crop_shape[0][0])
    crop_h_end = int(updated_config.training_data.crop_shape[0][1])
    offsets = deque(maxlen=5)
    last_offset = None
    expected_size = (int(updated_config.training_data.features.X.W), int(updated_config.training_data.features.X.H))
    for topic, msg, t in missync_bag.read_messages():
        # print(f"{t}: {topic}")

        if topic == '/sensor/velodyne/cloud_euclidean':
            previous_lidar_frames.append(deepcopy(msg))
            licas3_bag.write(topic, msg, t)
        elif topic == '/sensor/camera/grayscale/left/image_rect':
            previous_grayscale_frames.append(deepcopy(msg))
        elif topic == '/sensor/camera/color/left/image_rect':
            if len(previous_color_frames) < l:
                previous_color_frames.append(deepcopy(msg))
                licas3_bag.write('/sensor/velodyne/cloud_euclidean', previous_lidar_frames[-1], t)
                licas3_bag.write('/sensor/camera/color/left/image_rect', msg, t)
                licas3_bag.write('/sensor/camera/grayscale/left/image_rect', previous_grayscale_frames[-1], t)
            else:
                previous_color_frames.append(deepcopy(msg))
                # Run inference
                pcd_msg_copy = deepcopy(previous_lidar_frames[-1])
                pcd_msg_copy.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2  # https://github.com/eric-wieser/ros_numpy/issues/2
                pc = ros_numpy.numpify(pcd_msg_copy)
                pts_xyz = np.zeros((pc.shape[0], 3))
                pts_xyz[:, 0] = pc['x']
                pts_xyz[:, 1] = pc['y']
                pts_xyz[:, 2] = pc['z']
                X = None
                X_dense_depth_map_data = projection.get_dense_depth_map(
                    pts_xyz=pts_xyz,
                    H=camera_sensor_H,
                    W=camera_sensor_W,
                    T=T,
                    R=R,
                    P=P,
                    datasets_name=dataset_name,
                    kernel_size=8,
                    norm_methods=updated_config.training_data.z_norm_methods,
                    lidar_range=updated_config.sensors_info.lidar.range
                )
                # print("X_dense_depth_map_data: {}".format(X_dense_depth_map_data.shape))
                X_dense_depth_map_data_size = (X_dense_depth_map_data.shape[1], X_dense_depth_map_data.shape[0])
                X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
                X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size),
                                                        axis=-1)  # Resize
                for idx in range(1, l + 2):
                    camera_msg_copy = deepcopy(previous_color_frames[-idx])
                    camera_msg_copy.__class__ = sensor_msgs.msg.Image  # https://github.com/eric-wieser/ros_numpy/issues/2
                    X_camera_data = ros_numpy.numpify(camera_msg_copy)
                    X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                    X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                    X_camera_data = cv2.resize(X_camera_data, expected_size)
                    # print("X_camera_data: {}".format(X_camera_data.shape))
                    _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)
                    if X is None:
                        X = _X
                    else:
                        X = np.concatenate([X, _X], -1)
                X = [X / 255.]
                pred_synced_cam_idx = evaluator.inference([X])
                pred_synced_cam_idx = np.argmax(pred_synced_cam_idx)
                pred_synced_cam_idx_from_right = int(-(pred_synced_cam_idx + 1))
                offsets.append(pred_synced_cam_idx_from_right)
                averaged_offset = Counter(offsets).most_common(1)[0][0]
                print(f"({averaged_offset}) Offsets: {offsets}")

                if last_offset is None:
                    last_offset = averaged_offset
                if averaged_offset + offset_cnt <= last_offset + 1:
                    proposed_offset = last_offset
                    offset_cnt += 1
                else:
                    proposed_offset = averaged_offset
                    offset_cnt = 0

                last_offset = proposed_offset
                print("proposed_offset: {}".format(proposed_offset))
                previous_lidar_frames[proposed_offset].header.stamp = t
                previous_color_frames[proposed_offset].header.stamp = t
                previous_grayscale_frames[proposed_offset].header.stamp = t
                licas3_bag.write('/sensor/velodyne/cloud_euclidean', previous_lidar_frames[proposed_offset], t)
                licas3_bag.write('/sensor/camera/color/left/image_rect', previous_color_frames[proposed_offset], t)
                licas3_bag.write('/sensor/camera/grayscale/left/image_rect', previous_grayscale_frames[proposed_offset],
                                 t)

        elif 'color_labels' in topic:
            continue
        else:
            licas3_bag.write(topic, msg, t)

    missync_bag.close()
    licas3_bag.close()


def create_lidar_missync_rosbag(bag_fp, max_shift=10, shift_chance=0.5):
    """
    this function read a rosbag and create a synthetic rosbag with mis-synchronization
    :param bag_fp: original bagpath
    :param max_shift: max frames of shift
    :param shift_chance: the probability to do a shift
    :return:
    """
    # Hard coded params
    LATENCY_UPPER_BOUND = 10e7
    LIDAR_TOPIC = '/sensor/velodyne/cloud_euclidean'

    print("Reading bag ... {}".format(bag_fp))
    bag = rosbag.Bag(bag_fp)
    missync_bag = rosbag.Bag(f"{bag_fp}.lidar_missync.bag", 'w')
    latency = 0
    previous_frames = deque(maxlen=int(max_shift))
    cnt = 0
    for topic, msg, t in bag.read_messages():
        if topic == LIDAR_TOPIC:
            if len(previous_frames) == 0:
                previous_frames.append(deepcopy(msg))
            if random.random() > float(shift_chance):
                latency += int(LATENCY_UPPER_BOUND * random.random())  # 500 ms
                latency = min(latency, LATENCY_UPPER_BOUND * max_shift)
            else:
                latency -= int(LATENCY_UPPER_BOUND * random.random())  # 500 ms
                latency = max(latency, 0)
            frame_shift = int(latency / LATENCY_UPPER_BOUND)
            print("[{}] frame_shift: {}".format(cnt, frame_shift))
            if frame_shift >= 1:
                delayed_msg = previous_frames[-frame_shift]
                delayed_msg.header = msg.header
                missync_bag.write(topic, delayed_msg, t)
                print("Latency({} sec = frameshift = {})".
                      format(latency, frame_shift))
            else:
                missync_bag.write(topic, msg, t)
            previous_frames.append(deepcopy(msg))
            cnt += 1
        else:
            missync_bag.write(topic, msg, t)
    bag.close()
    missync_bag.close()


if __name__ == '__main__':
    fire.Fire()
