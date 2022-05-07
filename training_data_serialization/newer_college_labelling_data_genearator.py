from box import Box
import cv2
import fire
import multiprocessing
import numpy as np
import open3d as o3d
import os
import pandas as pd
import pathlib
import pickle
import random
import tensorflow as tf
from tqdm import tqdm
import yaml
from data_generator_base import Generator
from utils import projection
import uuid


class NewerCollegeGenerator(Generator):
    """
    ├── infra1
        ├── infra1_1583836591_152386717.png
        ├── ....
    ├── ouster_scan
        ├── cloud_1583836591_182590976.pcd
        ├── ....
    └── timeoffset
        ├── nc-long-time-offsets.csv
        ├── nc-moving-people-time-offsets.csv
        ├── nc-parkland-mound-time-offsets.csv
        ├── nc-quad-with-dynamics-time-offsets.csv
        ├── nc-short-time-offsets.csv
        └── nc-spinning-time-offsets.csv
    """

    @staticmethod
    def get_calibration_mtx(calibrations, sensors_info):
        h, w = sensors_info.camera.shape
        R_hom = np.array(calibrations.R_hom)
        P = np.array(calibrations.P)
        lidar_to_lidar_imu_hom = np.array(calibrations.lidar_to_lidar_imu_hom)
        lidar_imu_to_cam_hom = np.array(calibrations.lidar_imu_to_cam_hom)
        cam_intrinsics_mtx, roi = cv2.getOptimalNewCameraMatrix(R_hom, P, (w, h), 1, (w, h))
        lidar_to_cam = np.dot(lidar_to_lidar_imu_hom, lidar_imu_to_cam_hom.T).T
        return cam_intrinsics_mtx, lidar_to_cam

    @staticmethod
    def offset_with_timestamp(ts, offsets):
        """
        :param ts: a sorted array of ts, such as 1583836591.1523867
        :param offsets: csv file
                %time    field.header.seq   field.header.stamp   field.header.frame_id  field.timeOffset
        0   1583836591137266441   0         1583836621196352959   NaN                   0.056011
        1   1583836591137266441   1         1583836651489880085   NaN                   0.055845
        :return:
        """
        ts_with_offset = []
        cursor = 0
        offset_start = offsets["%time"][cursor]
        offset_end = offsets["field.header.stamp"][cursor]
        for _ts in ts:
            print(_ts)
            if offset_start > _ts:
                raise Exception("Offset datasets seem problematic!")
            if offset_start <= _ts <= offset_end:
                print("{} ~ {}: {}".format(offset_start, offset_end, offsets["field.timeOffset"][cursor]))
                ts_with_offset.append(_ts - offsets["field.timeOffset"][cursor])

            if _ts > offset_end:
                offset_start = offsets["field.header.stamp"][cursor]
                if cursor >= len(offsets["field.timeOffset"]) - 1:
                    # TODO: This data seems not proividing the offsets. Thus we discard
                    # ts_with_offset.append(_ts)
                    continue
                cursor += 1
                offset_end = offsets["field.header.stamp"][cursor]
                if offset_start <= _ts <= offset_end:
                    print("{} ~ {}".format(offset_start, offset_end))
                    ts_with_offset.append(_ts - offsets["field.timeOffset"][cursor])
                else:
                    raise Exception("Offset datasets seem problematic!")
        return ts_with_offset

    def get_raw_data_info(self, data_fp, data_info, lidar_sub_folder, camera_sub_folder, time_offsets_csv_fp):
        interested_folders = [lidar_sub_folder, camera_sub_folder]
        for root, dirs, files in os.walk(data_fp):
            for _dir in dirs:
                if _dir not in interested_folders:
                    print("Folder {} not interested, skip!".format(_dir))
                    continue
                _sub_folder = "{}/{}".format(root, _dir)
                for _subroot, _subdirs, _subfiles in os.walk(_sub_folder):
                    for _subfile in _subfiles:
                        _fp = "{}/{}".format(_subroot, _subfile)
                        _ts_info = _subfile.split("_")
                        _ts = "{}{}".format(_ts_info[1], _ts_info[2].split(".")[0])
                        if "(1)" in _ts:
                            _ts = _ts.split("(1)")[0]
                        data_info["sorted_{}_ts".format(_dir)].append(int(_ts))
                        data_info[_dir]["{}".format(int(_ts))] = _fp

        for _sub_folder in interested_folders:
            data_info["sorted_{}_ts".format(_sub_folder)] = np.sort(data_info["sorted_{}_ts".format(_sub_folder)])

        data_info["time_offsets"] = pd.read_csv("{}/{}".format(data_fp, time_offsets_csv_fp))
        data_info["sorted_{}_ts_after_offset".format(camera_sub_folder)] = \
            self.offset_with_timestamp(ts=data_info["sorted_{}_ts".format(camera_sub_folder)], offsets=data_info["time_offsets"])
        return data_info

    @staticmethod
    def sync_data(data_info, lidar_sub_folder, camera_sub_folder):
        sorted_ouster_scan_ts = data_info["sorted_{}_ts".format(lidar_sub_folder)]
        sorted_infra1_ts_after_offset = data_info["sorted_{}_ts_after_offset".format(camera_sub_folder)]
        data_info['synced_data_paris'] = {
            'paired_fp_seq': [],
            'paired_ts_seq': [],
            'paired_ts_diff': []
        }
        last_cursor = 0

        for _idx, _ouster_scan_ts in enumerate(sorted_ouster_scan_ts):
            _paired_ts = []
            _paired_fs = []
            _min_diff = np.inf
            for _cursor, _infra1_ts_after_offset in enumerate(sorted_infra1_ts_after_offset):
                if _cursor < last_cursor:
                    continue
                _diff = abs(_ouster_scan_ts - _infra1_ts_after_offset)
                if _diff <= _min_diff:
                    _min_diff = _diff
                    last_cursor += 1
                    _paired_ts = [_ouster_scan_ts, _infra1_ts_after_offset]
                    _paired_fs = [
                        data_info[lidar_sub_folder]["{}".format(_ouster_scan_ts)],
                        data_info[camera_sub_folder]["{}".format(data_info["sorted_{}_ts".format(camera_sub_folder)][_cursor])]
                    ]
                else:
                    continue
            if len(_paired_ts) > 0:
                data_info['synced_data_paris']['paired_fp_seq'].append(_paired_fs)
                data_info['synced_data_paris']['paired_ts_seq'].append(_paired_ts)
                data_info['synced_data_paris']['paired_ts_diff'].append(_min_diff)

        return data_info


    def get_training_examples_data_info(self, config, data_info_synced, visualization=False):
        """
        :param config:
        :param data_info_synced:
        data_info = {
                lidar_sub_folder: {},
                camera_sub_folder: {},
                "sorted_{}_ts".format(lidar_sub_folder): [],
                "sorted_{}_ts".format(camera_sub_folder): [],
                "sorted_{}_ts_after_offset".format(camera_sub_folder): [],
                "time_offsets": None,
                "synced_data_paris" :
                {
                    'paired_fp_seq': [],
                    'paired_ts_seq': [],
                    'paired_ts_diff': []
                }
        }
        :return:
        """
        # Print example of paired_fp_seq and paired_ts_seq
        # print(data_info_synced["synced_data_paris"]["paired_fp_seq"][:5])
        # print(data_info_synced["synced_data_paris"]["paired_ts_seq"][:5])
        """
        Sorted!
        [
            [
                '/media/kaiwen/extended/new_college/raw_data/ouster_scan/cloud_1583836591_182590976.pcd', 
                '/media/kaiwen/extended/new_college/raw_data/infra1/infra1_1583836591_185609553.png'
            ],
             [
                '/media/kaiwen/extended/new_college/raw_data/ouster_scan/cloud_1583836591_282592512.pcd', 
                '/media/kaiwen/extended/new_college/raw_data/infra1/infra1_1583836591_285496294.png'
            ]
        ]
        [[1583836591182590976, 1.5838365911856095e+18], [1583836591282592512, 1.5838365912854963e+18]]
        """
        datasets_name = config.name
        sampling_stride = int(config.training_data.sampling_stride)
        sampling_window = int(config.training_data.sampling_window)
        print("Sliding window is {}".format(sampling_window)) # 6
        training_examples_data_info = []
        R, T = self.get_calibration_mtx(config.calibrations, config.sensors_info)
        for _seq, _paired_frame in enumerate(data_info_synced["synced_data_paris"]["paired_fp_seq"]):
            if _seq < int(config.training_data.skip_frames):
                continue
            cnt = 0
            # (-6, 1) -> (-6, -5, -4, -3, -2, -1, 0)
            #         cnt >>
            # [camera, camera, camera], ... , [camera, camera, camera] : sampling_window
            # ...,                          >>>lidar<<<<,            lidar , lidar, ....
            for _i in range(- sampling_window * sampling_stride, 1, sampling_stride):
                if _seq + _i < 0:
                    continue
                # TODO: consider moving those hard-coded keys to config
                start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride
                end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1
                label_rel_idx = sampling_window - cnt
                example = {
                    'x.lidar.fp': _paired_frame[0],
                    'y.camera.fp': _paired_frame[1],
                    'x.camera.fps': [x[1] for x in data_info_synced["synced_data_paris"]["paired_fp_seq"][start_cam_idx : end_cam_idx: sampling_stride]],
                    'y.label': label_rel_idx,
                    'seq_id': _seq
                }
                if visualization:
                    pts = o3d.io.read_point_cloud(example["x.lidar.fp"])
                    pts_xyz = np.asarray(pts.points)
                    overlay_gt = projection.display_projected_img(pts_xyz, example["y.camera.fp"], T, R, datasets_name=datasets_name)
                    for _x_camera_fp in example["x.camera.fps"]:
                        print(example)
                        overlay_offset = projection.display_projected_img(pts_xyz, _x_camera_fp, T, R, datasets_name=datasets_name)
                        overlay = np.concatenate([overlay_offset, overlay_gt], 1)
                        cv2.imshow("overlay_gt{}".format(example['y.label']), overlay)
                        cv2.waitKey(0)

                training_examples_data_info.append(example)
                cnt += 1
        if visualization:
            cv2.destroyAllWindows()
        print("Generated {} training examples".format(len(training_examples_data_info)))
        # Generated 102515 training examples
        return training_examples_data_info

    @staticmethod
    def write_text(img, text, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 1
        fontColor = (0, 0, 255)
        lineType = 2

        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate_labelling_data_with_one_chunk_and_save_to_disk(self, config, output_fp, chunk_idx, data_chunk,
                                                                     camera_sensor_H, camera_sensor_W):
        if not os.path.isdir(output_fp):
            print("{} does not exits, creating one.".format(output_fp))
            pathlib.Path(output_fp).mkdir(parents=True, exist_ok=True)
        print("Creating {} th chunk of the labelling data ...".format(chunk_idx))
        # writer = tf.python_io.TFRecordWriter("{}/{}.tfrecord".format(output_fp, chunk_idx))
        datasets_name = config.name
        crop_h_start = int(config.training_data.crop_shape[0][0])
        crop_h_end = int(config.training_data.crop_shape[0][1])
        for _idx, _one_raw_data in enumerate(data_chunk):
            example_dict = {}
            label = _one_raw_data['y.label']
            seq_id = _one_raw_data['seq_id']
            R, T = self.get_calibration_mtx(config.calibrations, config.sensors_info)
            pts = o3d.io.read_point_cloud(_one_raw_data['x.lidar.fp'])
            pts_xyz = np.asarray(pts.points)
            X_dense_depth_map_data, z_before_norm = projection.get_dense_depth_map(
                pts_xyz=pts_xyz,
                H=camera_sensor_H,
                W=camera_sensor_W,
                T=T,
                R=R,
                datasets_name=datasets_name,
                get_z_before_norm=True,
                norm_methods=config.training_data.z_norm_methods,
                lidar_range=config.sensors_info.lidar.range)
            cnt = 0
            
            if len(_one_raw_data['x.camera.fps']) != config.training_data.sampling_window + 1:
                print("It shall have {} data but only got {} instead.".format(
                    config.training_data.sampling_window + 1, len(_one_raw_data['x.camera.fps'])))
                continue
            X_dense_depth_map_data_size = (X_dense_depth_map_data.shape[1], X_dense_depth_map_data.shape[0])
            expected_size = (int(config.labelling_data.size.W), int(config.labelling_data.size.H))
            X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
            X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size), axis=-1)
            display = None
            paddings = None
            X_dense_depth_map_data_display = None
            for _idx_fp, _camera_fp in enumerate(_one_raw_data['x.camera.fps']):
                X_camera_data = cv2.imread(_camera_fp)
                X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                X_camera_data = cv2.resize(X_camera_data, expected_size)
                _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)
                if display is None:
                    X_dense_depth_map_data_display = cv2.applyColorMap(X_dense_depth_map_data.astype(np.uint8), cv2.COLORMAP_JET)
                    display = cv2.addWeighted(X_camera_data, 1, X_dense_depth_map_data_display, 0.5, 1)
                    display = self.write_text(display, '{}'.format(cnt), 30, 30)
                    paddings = np.zeros_like(display)
                    paddings = self.write_text(paddings, 'Pred: {}'.format(label), 30, 30)
                else:
                    _display = cv2.addWeighted(X_camera_data, 1, X_dense_depth_map_data_display, 0.5, 1)
                    _display = self.write_text(_display, '{}'.format(cnt), 30, 30)
                    display = np.concatenate([display, _display], 1)
                cnt += 1
            display = np.concatenate([display, paddings, paddings], 1)
            display_h, display_w, _ = display.shape
            display = np.concatenate([display[:, :int(display_w / 2)], display[:, int(display_w / 2):]], 0)
            if not config.debug_mode:
                cv2.imwrite("{}/{}_seqid_{}_{}_gt{}.png".format(output_fp, _idx, seq_id, uuid.uuid4().hex, label), display)
            else:
                cv2.imshow("new_college.png", display.astype(np.uint8))
                cv2.imwrite("{}/{}_seqid_{}_{}_gt{}.png".format(output_fp, _idx, seq_id, uuid.uuid4().hex, label), display)
                cv2.waitKey(0)
        if not config.debug_mode:
            print("Created {}.{}.chunk labelling data!".format(output_fp, chunk_idx))
        else:
            cv2.destroyAllWindows()

    def generate_labelling_data_and_save_to_disk(self, datasets, config, name):
        _data_chunks = self.chunks(lst=datasets, n=config.labelling_data.chunk_size)
        output_fp = "{}/{}".format(config.labelling_data.output_dir, name)
        print("Creating {} labelling data with each including {} examples............".format(
            len(datasets) / config.training_data.chunk_size, config.training_data.chunk_size))

        camera_sensor_H, camera_sensor_W = config.sensors_info.camera.shape
        jobs = []
        for _chunk_idx, _data_chunk in tqdm(enumerate(_data_chunks)):
            if config.debug_mode:
                self.generate_labelling_data_with_one_chunk_and_save_to_disk(config, output_fp, _chunk_idx, _data_chunk, camera_sensor_H, camera_sensor_W)
            else:
                output_fp = "{}{}/{}".format(config.labelling_data.output_dir, _chunk_idx, name)
                p = multiprocessing.Process(
                    target=self.generate_labelling_data_with_one_chunk_and_save_to_disk,
                    args=(config, output_fp, _chunk_idx, _data_chunk, camera_sensor_H, camera_sensor_W))
                jobs.append(p)
                p.start()

    def generate_training_data(self, config, re_sync, *args, **kwargs):
        """
        example = {
            'x.lidar.fp': _paired_frame[0],
            'y.camera.fp': _paired_frame[1],
            'x.camera.fps': [x[1] for x in data_info_synced["synced_data_paris"]["paired_fp_seq"][_seq + _i: _seq + _i + sampling_window + 1]],
            'y.label': -_i
        }
        :param config:
        :param args:
        :param kwargs:
        :return:
        """
        print("Configs: \n{}".format(config))

        camera_sub_folder = config.raw_data.camera_sub_folder
        lidar_sub_folder = config.raw_data.lidar_sub_folder
        time_offsets_csv_fp = config.raw_data.time_offsets_csv_fp

        _synced_raw_data_info = config.raw_data.generated_fp.synced_raw_data_info
        if re_sync:
            print("Step1: scan folder to construct data_info: raw data not loaded ...")
            data_info = {
                lidar_sub_folder: {},
                camera_sub_folder: {},
                "sorted_{}_ts".format(lidar_sub_folder): [],
                "sorted_{}_ts".format(camera_sub_folder): [],
                "sorted_{}_ts_after_offset".format(camera_sub_folder): [],
                "time_offsets": None
            }
            data_info = self.get_raw_data_info(
                data_fp=config.raw_data.root_dir,
                data_info=data_info,
                camera_sub_folder=camera_sub_folder,
                lidar_sub_folder=lidar_sub_folder,
                time_offsets_csv_fp=time_offsets_csv_fp
            )
            print("Step2: synchronize the data: raw data not loaded...")
            data_info_synced = self.sync_data(
                data_info=data_info,
                camera_sub_folder=camera_sub_folder,
                lidar_sub_folder=lidar_sub_folder
            )
            print("Step3: Save the synced data info somewhere to avoid do it again...")
            with open(_synced_raw_data_info, 'wb') as handle:
                pickle.dump(data_info_synced, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Data sync-ed process skip, directly loads from the pickle")
            with open(_synced_raw_data_info, 'rb') as handle:
                data_info_synced = pickle.load(handle)

        training_examples_data_info = self.get_training_examples_data_info(config, data_info_synced,
                                                                           visualization=False)
        return training_examples_data_info

    @staticmethod
    def down_sample(data_info, downsample_ratio):
        return random.sample(data_info, int(float(downsample_ratio) * len(data_info)))

    def generate_labelling_data(self, config, training_examples_data_info, *args, **kwargs):
        # Step1: Split the data into training/validation/testing
        sampling_window = int(config.training_data.sampling_window)
        sampling_stride = int(config.training_data.sampling_stride)
        padding_number = sampling_window * sampling_stride
        # due to the way that we generate the training data,
        # to avoid any observations occur in the validation/testing datasets, we add 20 in between
        # namely:
        # 0 ,....., 71760 th: training
        # 71760 + 21 th, ..... 92263 th: validation
        # 92263 + 21 th, .... 102515 th: testing
        training_data_idx = int(float(config.training_data.split_ratio[0] * len(training_examples_data_info)))
        validation_data_idx = int(
            (float(config.training_data.split_ratio[0]) + float(config.training_data.split_ratio[1])) * len(
                training_examples_data_info))
        print("Training data will be ranging from {} to {} ".format(0, training_data_idx))
        print("Validation data will be ranging from {} to {} ".format(training_data_idx, validation_data_idx))
        print(
            "Testing data will be ranging from {} to {}".format(validation_data_idx, len(training_examples_data_info)))
        training_data_examples_data_info = training_examples_data_info[0:training_data_idx]
        validation_data_examples_data_info = training_examples_data_info[
                                             training_data_idx + padding_number: validation_data_idx]
        testing_data_examples_data_info = training_examples_data_info[validation_data_idx + padding_number:]

        # Step2: Down sample the data
        training_data_examples_data_info = self.down_sample(training_data_examples_data_info,
                                                            config.labelling_data.downsample_ratio)
        validation_data_examples_data_info = self.down_sample(validation_data_examples_data_info,
                                                              config.labelling_data.downsample_ratio)
        testing_data_examples_data_info = self.down_sample(testing_data_examples_data_info,
                                                           config.labelling_data.downsample_ratio)
        print("Summary:")
        print("Training examples: {}".format(len(training_data_examples_data_info)))
        print("Validation examples: {}".format(len(validation_data_examples_data_info)))
        print("Testing examples: {}".format(len(testing_data_examples_data_info)))
        # Training examples: 35880
        # Validation examples: 10241
        # Testing examples: 5115
        print("Testing data start from: {}".format(testing_data_examples_data_info[0]))
        # self.generate_labelling_data_and_save_to_disk(training_data_examples_data_info, config, "training")
        # self.generate_labelling_data_and_save_to_disk(validation_data_examples_data_info, config, "validation")
        self.generate_labelling_data_and_save_to_disk(testing_data_examples_data_info, config, "testing")

    def run(self, config_fp, re_sync, *args, **kwargs):
            print("Step1: Loading configuration file ...")
            config = Box(yaml.load(open(config_fp, 'r').read()))

            print("Step2: Generate training data ...")
            training_examples_data_info = self.generate_training_data(config=config, re_sync=re_sync)

            print("Step3: Serialize data into tfrecords ...")
            self.generate_labelling_data(config=config, training_examples_data_info=training_examples_data_info)


if __name__ == '__main__':
    fire.Fire(NewerCollegeGenerator)
