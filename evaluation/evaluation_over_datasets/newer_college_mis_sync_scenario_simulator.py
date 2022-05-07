'''
This module is to generate the newer college demo with the psudo ground truth
'''
import cv2
import fire
import numpy as np
from utils import pb_server
from utils import projection
from utils.metrics import print_stats
import open3d as o3d
import pandas as pd
import os
import pickle
import random
from tqdm import tqdm
from copy import deepcopy


class NewerCollegeMisSyncScenarioSimulator(object):

    def batch_run_over_psudo_gt(self, config, *args, **kwargs):
        print("Step1: directly get config ...")
        config = config

        print("Step2: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step3: Generate Demo ...")
        return self.generate_demos(config=config, evaluator=evaluator)

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
            self.offset_with_timestamp(ts=data_info["sorted_{}_ts".format(camera_sub_folder)],
                                       offsets=data_info["time_offsets"])
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
                        data_info[camera_sub_folder][
                            "{}".format(data_info["sorted_{}_ts".format(camera_sub_folder)][_cursor])]
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
        num_frames_in_seq = int(config.training_data.multi_frame_test.num_frames_in_seq)
        datasets_name = config.name
        sampling_window = int(config.training_data.sampling_window)
        sampling_stride = int(config.training_data.sampling_stride)
        print("Sliding window is {}".format(sampling_window)) # 6
        training_examples_data_info = []
        R, T = self.get_calibration_mtx(config.calibrations, config.sensors_info)
        proposed_end_cam_idxs = []
        simulated_drop_packet = 0
        for _seq, _paired_frame in enumerate(data_info_synced["synced_data_paris"]["paired_fp_seq"]):
            if _seq < int(config.training_data.skip_frames):
                continue
            # (-6, 1) -> (-6, -5, -4, -3, -2, -1, 0)
            #         cnt >>
            # [camera, camera, camera], ... , [camera, camera, camera] : sampling_window
            # ...,                          >>>lidar<<<<,            lidar , lidar, ....
            _i = random.choice(range(- sampling_window * sampling_stride, 1, sampling_stride))
            if _seq + _i < 0:
                continue
            if (_seq + num_frames_in_seq * sampling_stride) >= len(data_info_synced["synced_data_paris"]["paired_fp_seq"]):
                continue

            cnt = range(- sampling_window * sampling_stride, 1, sampling_stride).index(_i)
            # Below logics is to avoid backwards offset
            proposed_start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride + (num_frames_in_seq - 1) * sampling_stride
            proposed_end_cam_idx = proposed_start_cam_idx + sampling_window * sampling_stride + 1
            if simulated_drop_packet < num_frames_in_seq * sampling_stride + 2:
                simulated_drop_packet += 1
                continue
            if (len(proposed_end_cam_idxs) > 0) and (proposed_end_cam_idx < max(proposed_end_cam_idxs)):
                print("Skip.........................")
                simulated_drop_packet += 1
                continue
            simulated_drop_packet = 0
            proposed_end_cam_idxs.append(proposed_end_cam_idx)
            # Above logics is to avoid backwards offset -- uncomment to activate
            label_rel_idx = sampling_window - cnt
            examples = []
            for _sub_seq in range(0, num_frames_in_seq):
                start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride + _sub_seq * sampling_stride
                end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1

                example = {
                    'x.lidar.fp': data_info_synced["synced_data_paris"]["paired_fp_seq"][_seq+_sub_seq * sampling_stride][0],
                    'y.camera.fp': data_info_synced["synced_data_paris"]["paired_fp_seq"][_seq+_sub_seq * sampling_stride][1],
                    'x.camera.fps': [x[1] for x in data_info_synced["synced_data_paris"]["paired_fp_seq"][
                                                   start_cam_idx: end_cam_idx: sampling_stride]],
                    'drive': 'test',
                    'y.label': label_rel_idx,
                    'seq_id': _seq
                }
                examples.append(example)
            training_examples_data_info.append(examples)
        print("Generated {} training examples".format(len(training_examples_data_info)))
        # Generated 102515 training examples
        return training_examples_data_info

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

    def generate_demos(self, config, evaluator):

        print("     Step1: Generate inference data ...")
        inference_data_generator = self.generate_inference_data(config=config)
        metrics = {
            'total': {
                'true': 0,
                'false': 0,
                'true-neighbour': 0
            }
        }
        print("     Step2: Running benchmarks ...")
        cnt = 0
        for inference_data_seq in tqdm(inference_data_generator):
            preds = []
            Y_gt = None
            drive_number = None
            if inference_data_seq is None:
                continue
            if len(inference_data_seq) < int(config.training_data.multi_frame_test.num_frames_in_seq):
                continue
            for dm_idx, inference_data in enumerate(inference_data_seq):
                inf_h, inf_w, _ = inference_data["X"].shape
                X = [inference_data["X"] / 255.]
                Y = inference_data["Y"]
                [pred_synced_frame] = evaluator.inference([X])
                pred_synced_frame = np.argmax(pred_synced_frame)
                if drive_number not in metrics:
                    metrics[drive_number] = {
                        'true': 0,
                        'false': 0,
                        'true-neighbour': 0
                    }
                if Y_gt is None:
                    Y_gt = Y
                else:
                    if Y_gt != Y:
                        raise Exception("Y_gt != Y")
                preds.append(int(pred_synced_frame))
            pred = np.bincount(preds).argmax()
            print("Pred: {} ~> {} | Gt: {}".format(preds, pred, Y_gt))
            #  Write example:
            display = inference_data_seq[0]["display"]
            display_h, display_w, _ = display.shape
            display_unit_w = inf_w * 3
            test_id = inference_data_seq[0]["test_id"]
            pred_display = deepcopy(display[:, display_unit_w*(int(pred)+1):display_unit_w*(int(pred)+2),:])
            gt_display = deepcopy(display[:, :display_unit_w, :])
            display_h, display_w, _ = display.shape

            num_seq = int(display_w / display_unit_w) - 1
            display_to_write = display[:, display_unit_w:display_unit_w * 2, :]
            for i in range(2, num_seq + 1):
                display_to_write = np.concatenate(
                    [display_to_write, display[:, display_unit_w * i:display_unit_w * (i + 1), :]], 1)
            display_to_write = np.concatenate([display_to_write, pred_display, gt_display], 1)

            cv2.imwrite("simulation_video/newer_college/{}_{}_pred@{}.png".format(test_id, Y_gt, pred), display_to_write)

            #  Write example:
            if int(pred) == Y_gt:
                metrics['total']['true'] += 1
                metrics[drive_number]['true'] += 1

            else:
                metrics['total']['false'] += 1
                metrics[drive_number]['false'] += 1
                if int(abs(pred - Y_gt)) == 1:
                    metrics['total']['true-neighbour'] += 1
                    metrics[drive_number]['true-neighbour'] += 1

            print_stats(metrics)
            cnt += 1
        print(metrics)
        print_stats(metrics)
        return metrics

    @staticmethod
    def write_text(img, text, x, y, fontColor=(0, 0, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 0.6
        lineType = 1

        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    def generate_inference_data(self, config, re_sync=False, *args, **kwargs):
        """
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
        training_examples_data_info = self.generate_training_data(config=config, re_sync=re_sync)
        print("Seq ID ranging from {} to {}".format(training_examples_data_info[0][0]['seq_id'], training_examples_data_info[-1][0]['seq_id']))
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
        print("Before Down-sampling:")
        print("Training examples: {}".format(len(training_data_examples_data_info)))
        print("Validation examples: {}".format(len(validation_data_examples_data_info)))
        print("Testing examples: {}".format(len(testing_data_examples_data_info)))
        print("=============================================================================")
        print("Testing Seq ID ranging from {} to {}".format(testing_data_examples_data_info[0][0]['seq_id'], testing_data_examples_data_info[-1][0]['seq_id']))
        print("Testing data start from: {}".format(testing_data_examples_data_info[0]))
        datasets_name = config.name
        camera_sensor_H, camera_sensor_W = config.sensors_info.camera.shape
        crop_h_start = int(config.training_data.crop_shape[0][0])
        crop_h_end = int(config.training_data.crop_shape[0][1])
        data_info_synced = testing_data_examples_data_info
        for _idx_seq, _one_raw_data_maybe_a_seq in enumerate(data_info_synced):
            example_dicts = []
            for _, _one_raw_data in enumerate(_one_raw_data_maybe_a_seq):
                Y = _one_raw_data['y.label']
                seq_id = _one_raw_data['seq_id']
                test_id = "{}_{}".format(seq_id, Y)
                R, T = self.get_calibration_mtx(config.calibrations, config.sensors_info)
                updated_fp = '{}{}'.format(config.raw_data.root_dir, _one_raw_data['x.lidar.fp'].split('raw_data')[-1])
                pts = o3d.io.read_point_cloud(updated_fp)
                pts_xyz = np.asarray(pts.points)
                X_dense_depth_map_data = projection.get_dense_depth_map(
                    pts_xyz=pts_xyz,
                    H=camera_sensor_H,
                    W=camera_sensor_W,
                    T=T,
                    R=R,
                    datasets_name=datasets_name,
                    norm_methods=config.training_data.z_norm_methods,
                    lidar_range=config.sensors_info.lidar.range
                )
                cnt = 0
                if len(_one_raw_data['x.camera.fps']) != config.training_data.sampling_window + 1:
                    print("It shall have {} data but only got {} instead.".format(
                        config.training_data.sampling_window + 1, len(_one_raw_data['x.camera.fps'])))
                    continue

                X_dense_depth_map_data_size = (X_dense_depth_map_data.shape[1], X_dense_depth_map_data.shape[0])
                expected_size = (int(config.training_data.features.X.W), int(config.training_data.features.X.H))
                expected_display_size = (
                int(config.training_data.features.X.W) * 3, int(config.training_data.features.X.H) * 5)
                X = None
                display = None
                X_dense_depth_map_data_display = np.expand_dims(
                    cv2.resize(deepcopy(X_dense_depth_map_data), expected_display_size),
                    axis=-1)  # Resize
                X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
                X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size), axis=-1)

                for _idx_fp, _camera_fp_origin in enumerate(_one_raw_data['x.camera.fps']):
                    cnt += 1

                    _camera_fp_info = _camera_fp_origin.split('new_college/raw_data')
                    _camera_fp = "{}{}".format(config.raw_data.root_dir, _camera_fp_info[-1])
                    X_camera_data = cv2.imread(_camera_fp)
                    X_camera_data_display = deepcopy(X_camera_data)
                    X_camera_data_display = cv2.resize(X_camera_data_display, expected_display_size)

                    X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                    X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                    X_camera_data = cv2.resize(X_camera_data, expected_size)
                    _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)

                    if display is None:
                        X_dense_depth_map_data_display = cv2.applyColorMap(X_dense_depth_map_data_display.astype(np.uint8),
                                                                           cv2.COLORMAP_MAGMA)
                        display = cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display, 0.5, 1)
                    else:
                        display = np.concatenate(
                            [display, cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display, 0.5, 1)], 1)
                    if _idx_fp == Y:
                        display_gt = cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display, 0.5, 1)

                    if X is None:
                        X = _X
                    else:
                        X = np.concatenate([X, _X], -1)
                display = np.concatenate([display_gt, display], 1)
                example_dict = {
                    config.training_data.features.X.feature_name: X.astype(np.float32),
                    config.training_data.features.Y.feature_name: Y,
                    "test_id": test_id,
                    'drive': 'test',
                    'display': display

                }
                example_dicts.append(example_dict)
            yield example_dicts


if __name__ == '__main__':
    fire.Fire(NewerCollegeMisSyncScenarioSimulator)
