"""
This module is all about loading raw data from KITTI
--------------------------------------------------------------
Some descriptions on KITTI Raw Data.
    1. Raw data can be downloaded from http://www.cvlibs.net/datasets/kitti/raw_data.php
    2. Download one of examples "2011_09_26_drive_0001" by running data/kitti_example_raw_data_download.sh
       You will see folder structure as below:
       - 2011_09_26
               |- 2011_09_26_drive_0001_sync
                    |- image_00
                        |- data
                            |- 0000000000.png (1242 x 375)
                            |- ......
                        |- timestamps.txt (don't care in this project)
                    |- image_01
                        ...
                    |- image_02
                        ...
                    |- image_03
                        ...
                    |- oxts (don't care in this project)
                    |- velodyne_points
                        |- data
                            |- 0000000000.bin
                            |- ....
                        |- timestamps.txt
                        |- timestamps_end.txt
                        |- timestamps_start.txt
               |- 2011_09_26_drive_0001_exact (similar structure as above except photo is 1382 x 512)
               |- calib_cam_to_cam.txt
               |- calib_imu_to_velo.txt
               |- calib_velo_to_cam.txt

    3.  All camera images are provided as lossless compressed and rectified png
        sequences. The native image resolution is 1382x512 pixels and a little bit
        less after rectification, for details see the calibration section below.
        The opening angle of the cameras (left-right) is approximately 90 degrees.

        The camera images are stored in the following directories:


        - 'image_00': left rectified grayscale image sequence
        - 'image_01': right rectified grayscale image sequence
        - 'image_02': left rectified color image sequence
        - 'image_03': right rectified color image sequence

        The sensor calibration zip archive contains files, storing matrices in
        row-aligned order, meaning that the first values correspond to the first
        row:

        calib_cam_to_cam.txt: Camera-to-camera calibration
        --------------------------------------------------

        - S_xx: 1x2 size of image xx before rectification
        - K_xx: 3x3 calibration matrix of camera xx before rectification
        - D_xx: 1x5 distortion vector of camera xx before rectification
        - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
        - T_xx: 3x1 translation vector of camera xx (extrinsic)
        - S_rect_xx: 1x2 size of image xx after rectification
        - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
        - P_rect_xx: 3x4 projection matrix after rectification

        Note: When using this dataset you will most likely need to access only
        P_rect_xx, as this matrix is valid for the rectified image sequences.

        calib_velo_to_cam.txt: Velodyne-to-camera registration
        ------------------------------------------------------

        - R: 3x3 rotation matrix
        - T: 3x1 translation vector
        - delta_f: deprecated
        - delta_c: deprecated

        R|T takes a point in Velodyne coordinates and transforms it into the
        coordinate system of the left video camera. Likewise it serves as a
        representation of the Velodyne coordinate frame in camera coordinates.

        calib_imu_to_velo.txt: GPS/IMU-to-Velodyne registration
        -------------------------------------------------------

        - R: 3x3 rotation matrix
        - T: 3x1 translation vector

        R|T takes a point in GPS/IMU coordinates and transforms it into the
        coordinate system of the Velodyne scanner. Likewise it serves as a
        representation of the GPS/IMU coordinate frame in Velodyne coordinates.

"""
import os
import pprint
import numpy as np


def get_raw_data_stats(data_fp, datatype='sync'):
    """
    This is a helper function that will scan KITTI raw data folder and output stats
    :param data_fp: absolute folder path
    :return:
        - stats: dictionary of stats
        - data_fps: baked dictionary of data_fps, e.g:
         {'2011_09_26_drive_0002_sync':
                {   'abs_path':  '~/../kitti_raw_data/2011_09_26/2011_09_26_drive_0002_sync',
                    'data_fps':
                        ['0000000034',
                        '0000000020',
                        .....
                        ]
                }
        }

    With this, you can construct the path for every single data with dictionary's value,
    given the facts that KITTI use png and bin for image and velodyne data and the number are the same.
    """
    stats = {
        "num_drive": 0,
        "num_data": 0,
        "num_data_stats": {}
    }
    data_fps = {}
    for root, dirs, files in os.walk(data_fp):
        if "velodyne_points" in dirs:
            path = root.split(os.sep)
            if datatype in path[-1]:
                data_fps[path[-1]] = {
                    "abs_path": root,
                    "data_ids": [],
                    "drive_name": path[-1]
                }
                stats["num_data_stats"][path[-1]] = 0
                stats["num_drive"] += 1
                img_lrc_folder = '{}/velodyne_points/data'.format(root)
                for _subroot, _subdirs, _files in os.walk(img_lrc_folder):
                    for _file in _files:
                        _file_num = _file.split('.')[0]
                        data_fps[path[-1]]["data_ids"].append(_file_num)
                        stats["num_data"] += 1
                        stats["num_data_stats"][path[-1]] += 1
    print("------------------------------Data Stats--------------------------")
    pprint.pprint(stats)
    print("------------------------------------------------------------------")
    return stats, data_fps


def parse_from_kitti_txt(data_fp):
    data = {}
    with open(data_fp, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            if 'delta' in key: continue
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_velo_to_cam_calibrations(data_fp):
    data = parse_from_kitti_txt(data_fp=data_fp)
    R_rm = np.reshape(data['R'], [3, 3])
    T = np.reshape(data['T'], [3, 1])
    ext_calib = np.concatenate((R_rm, T), axis=-1)
    return ext_calib


def load_cam_calibrations(data_fp):
    data = parse_from_kitti_txt(data_fp=data_fp)
    P_rect_02 = data['P_rect_02']
    R_rect_02 = data['R_rect_02']
    R_rect = np.reshape(R_rect_02, [3, 3])
    P_rect = np.reshape(P_rect_02, [3, 4])
    return P_rect, R_rect


def load_raw_data(data_fp):
    """
    Below we only import image_02, left rectified color image

    This is a helper function that will load all KITTI raw data into memory at once.
    :param data_fp: absolute folder path
    :return: a data info
    data_info = {
            "paired_fp_seq": {
                "2011_09_26_drive_0001_sync":
                [
                    [".../velodyne_points/data/0000000001.bin", ".../image_02/data/0000000001.png"],
                    ...
                    [".../velodyne_points/data/0000000284.bin", ".../image_02/data/0000000284.png"],
                ],
                ...
            },
            "T": None,
            "P": None,
            "R": None
        }
    """
    print("Loading raw data from {}.....".format(data_fp))
    stats, data_fps = get_raw_data_stats(data_fp=data_fp)
    data_drives = data_fps.values()
    transform_mat_fp = None
    transform_mat = None
    data_info = {
        "paired_fp_seq": {},
        "T": None,
        "P": None,
        "R": None
    }
    for _data_drive in data_drives:
        _abs_path = _data_drive["abs_path"]
        _data_ids = _data_drive["data_ids"]
        _data_ids = sorted(_data_ids)
        _data_drive_name = _data_drive["drive_name"]
        for _data_id in _data_ids:
            lidar_points_fp = '{}/velodyne_points/data/{}.bin'.format(_abs_path, _data_id)
            cam_points_fp = '{}/image_02/data/{}.png'.format(_abs_path, _data_id)
            if _data_drive_name not in data_info["paired_fp_seq"]:
                data_info["paired_fp_seq"][_data_drive_name] = []
            data_info["paired_fp_seq"][_data_drive_name].append([lidar_points_fp, cam_points_fp])
            _transform_mat_fp = '{}/../calib_velo_to_cam.txt'.format(_abs_path)
            _cam_mat_fp = '{}/../calib_cam_to_cam.txt'.format(_abs_path)

            # Below logic is to ensure you don't read and parse the same transform matrix
            # for every data. We only read for every drive even though as long as they are the same
            # day, they shall be the same.
            if (transform_mat is None) or (transform_mat_fp != _transform_mat_fp):
                transform_mat_fp = _transform_mat_fp
                transform_mat = load_velo_to_cam_calibrations(transform_mat_fp)
            # else:
            #     print("Skip reading transform matrix again, since it is the same.")
            # print _data_id
            data_info["T"] = transform_mat
            data_info["P"], data_info["R"] = load_cam_calibrations(_cam_mat_fp)
    print("{} raw data has been loaded!".format(len(data_info)))
    print("Example data look like below:")
    return data_info


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    load_raw_data(data_fp='{}/../kitti_raw_data'.format(dir_path))
