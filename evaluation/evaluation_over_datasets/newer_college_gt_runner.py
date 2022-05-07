'''
This module is to benchmark the newer college models with the real ground truth
'''
import cv2
import fire
import numpy as np
from utils import pb_server, projection
from utils.metrics import print_stats
import open3d as o3d
import pandas as pd
import os
import pickle
import random
from tqdm import tqdm


class NewerCollegeBenchmarksOverGT(object):

    def batch_run_over_gt(self, config, *args, **kwargs):
        print("Step1: directly get config ...")
        config = config

        print("Step2: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step3: Loading testing data ...")
        labelled_testing_data_dict = self.get_testing_with_labels(config)

        print("Step3: Run benchmarks ...")
        return self.run_benchmarks(config=config, evaluator=evaluator, labelled_testing_data_dict=labelled_testing_data_dict)

    def get_testing_with_labels(self, config):
        """
        :param config:
        :return:
        labelled_testing_data_dict =
        {
            "14330_5": {
                'gt': 1,
                'baseline': 4,
                'sub_folder_id': 94 # not really usefull, meta info
            },
            ...
        }
        """
        sub_folders = config.labelled_data.sub_folders
        labelled_testing_data_dict = {}
        for _sub_folder in sub_folders:
            data_dir = config.labelled_data.data_dir
            label_fp = config.labelled_data.label_fp
            testing_dir = data_dir.format(_sub_folder)
            label_fp = label_fp.format(_sub_folder)
            labels = pd.read_csv(label_fp)
            for _subroot, _subdirs, _subfiles in os.walk(testing_dir):
                for _subfile in _subfiles:
                    info = _subfile.split('_')
                    sub_folder_id = int(info[0])
                    seq_id = int(info[2])
                    baseline_pred = int(info[-1][2])
                    test_id = "{}_{}".format(seq_id, baseline_pred)
                    gt = labels.label[sub_folder_id]
                    if (gt == 'unsure') or (gt == 'disagreed'):
                        # print(">> Invalid testing example, skip")
                        continue
                    labelled_testing_data_dict[test_id] = {
                        'gt': gt,
                        'baseline': baseline_pred,
                        'sub_folder_id': sub_folder_id
                    }

        print("{} data has been labelled!".format(len(labelled_testing_data_dict.keys())))
        return labelled_testing_data_dict

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
        sampling_window = int(config.training_data.sampling_window)
        sampling_stride = int(config.training_data.sampling_stride)
        print("Sliding window is {}".format(sampling_window)) # 6
        training_examples_data_info = []
        R, T = self.get_calibration_mtx(config.calibrations, config.sensors_info)
        for _seq, _paired_frame in enumerate(data_info_synced["synced_data_paris"]["paired_fp_seq"]):
            if _seq < int(config.training_data.skip_frames):
                continue
            # (-6, 1) -> (-6, -5, -4, -3, -2, -1, 0)
            #         cnt >>
            # [camera, camera, camera], ... , [camera, camera, camera] : sampling_window
            # ...,                          >>>lidar<<<<,            lidar , lidar, ....
            for _i in range(- sampling_window * sampling_stride, 1, sampling_stride):
                if _seq + _i < 0:
                    continue
                cnt = range(- sampling_window * sampling_stride, 1, sampling_stride).index(_i)
                start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride
                end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1
                label_rel_idx = sampling_window - cnt
                example = {
                    'x.lidar.fp': _paired_frame[0],
                    'y.camera.fp': _paired_frame[1],
                    'x.camera.fps': [x[1] for x in data_info_synced["synced_data_paris"]["paired_fp_seq"][
                                                   start_cam_idx: end_cam_idx: sampling_stride]],
                    'drive': 'test',
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
        if visualization:
            cv2.destroyAllWindows()
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

    def run_benchmarks(self, config, evaluator, labelled_testing_data_dict):

        print("     Step1: Generate inference data ...")
        inference_data_generator = self.generate_inference_data(config=config)

        metrics = {
            'over_gt': {
                'true': 0,
                'false': 0,
                'true-neighbour': 0
            },
            'baseline_over_gt': {
                'true': 0,
                'false': 0,
                'true-neighbour': 0
            },
            'pred_to_psudo_gt': {
                'true': 0,
                'false': 0,
                'true-neighbour': 0
            }
        }
        check_list = {

        }
        print("     Step2: Running benchmarks ...")
        cnt = 0
        tested_test_ids = []
        for inference_data in tqdm(inference_data_generator):
            test_id = inference_data['test_id']
            if test_id not in labelled_testing_data_dict.keys():
                continue
            else:
                gt = labelled_testing_data_dict[test_id]['gt']
                gts_string = gt.split('/')
                gts = []
                for gt_string in gts_string:
                    if not (gt_string == ''):
                        gts.append(int(gt_string))
                tested_test_ids.append(test_id)
            X = [inference_data["X"] / 255.]
            Y = inference_data["Y"]
            viz = None
            [pred_synced_frame] = evaluator.inference([X], profile_export='newer_college')
            pred_synced_frame = np.argmax(pred_synced_frame)
            dms = np.concatenate(
                [inference_data["X"][:, :, [-1]], inference_data["X"][:, :, [-1]], inference_data["X"][:, :, [-1]]], -1)
            # if config.results.write_viz:
            #     for i in range(0, int(config.training_data.num_camera_frames) + 1):
            #         if viz is None:
            #             viz = np.squeeze(inference_data["X"][:, :, i * 4: (i * 4 + 3)]) + dms
            #         else:
            #             viz = np.concatenate([viz, dms + np.squeeze(inference_data["X"][:, :, i * 4: (i * 4 + 3)])], 0)
            #
            #     viz = np.concatenate([viz, dms], 0)
            #     # cv2.putText(viz,  inference_data['drive'], (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)
            #     # cv2.putText(viz,  inference_data['seq'], (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)
            #     cv2.putText(viz, "Pred: {}".format(int(pred_synced_frame)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.3,
            #                 (0, 0, 255), 1)
            #     cv2.putText(viz, "GT  : {}".format(Y), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
            #     cv2.imwrite("{}/new_college_seq1/{}.png".format(config.results.save_dir, cnt), viz)
            #     print("Writing data into {}/new_college_seq1/{}.png".format(config.results.save_dir, cnt))

            if int(pred_synced_frame) in gts:
                metrics['over_gt']['true'] += 1
            else:
                metrics['over_gt']['false'] += 1
                if np.min(np.abs(np.array(gts) - int(pred_synced_frame))) == 1:
                    metrics['over_gt']['true-neighbour'] += 1
                check_list[test_id] = {
                    'pred': int(pred_synced_frame)
                }
            print("[{} | {}] Prediction: {} | Baseline: {} | GTs: {}".format(cnt, test_id, pred_synced_frame, Y, gts))
            if Y in gts:
                metrics['baseline_over_gt']['true'] += 1
            else:
                metrics['baseline_over_gt']['false'] += 1
                if np.min(np.abs(np.array(gts) - int(pred_synced_frame))) == 1:
                    metrics['baseline_over_gt']['true-neighbour'] += 1

            if int(pred_synced_frame) == Y:
                metrics['pred_to_psudo_gt']['true'] += 1
            else:
                metrics['pred_to_psudo_gt']['false'] += 1
                if abs(int(pred_synced_frame) - Y) == 1:
                    metrics['pred_to_psudo_gt']['true-neighbour'] += 1

            print_stats(metrics)
            cnt += 1
        print(metrics)
        print_stats(metrics)
        print("{} has been tested.".format(metrics['over_gt']['true'] + metrics['over_gt']['false']))
        print("Tested test id number is {}".format(len(tested_test_ids)))
        print("Labelled test id number is {}".format(len(labelled_testing_data_dict.keys())))
        print(check_list)
        return metrics

    @staticmethod
    def down_sample(data_info, downsample_ratio):
        return random.sample(data_info, int(float(downsample_ratio) * len(data_info)))

    def generate_inference_data(self, config, visualization=False, re_sync=False, *args, **kwargs):
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
        print("Seq ID ranging from {} to {}".format(training_examples_data_info[0]['seq_id'], training_examples_data_info[-1]['seq_id']))
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
        print("Testing Seq ID ranging from {} to {}".format(testing_data_examples_data_info[0]['seq_id'], testing_data_examples_data_info[-1]['seq_id']))
        # Step2: Down sample the data
        # training_data_examples_data_info = self.down_sample(training_data_examples_data_info,
        #                                                     config.training_data.downsample_ratio)
        # validation_data_examples_data_info = self.down_sample(validation_data_examples_data_info,
        #                                                       config.training_data.downsample_ratio)
        # testing_data_examples_data_info = self.down_sample(testing_data_examples_data_info,
        #                                                    config.training_data.downsample_ratio)
        # print("Summary:")
        # print("Training examples: {}".format(len(training_data_examples_data_info)))
        # print("Validation examples: {}".format(len(validation_data_examples_data_info)))
        # print("Testing examples: {}".format(len(testing_data_examples_data_info)))
        # Training examples: 35880
        # Validation examples: 10241
        # Testing examples: 5115
        print("Testing data start from: {}".format(testing_data_examples_data_info[0]))
        datasets_name = config.name
        camera_sensor_H, camera_sensor_W = config.sensors_info.camera.shape
        crop_h_start = int(config.training_data.crop_shape[0][0])
        crop_h_end = int(config.training_data.crop_shape[0][1])
        data_info_synced = testing_data_examples_data_info
        for _idx, _one_raw_data in enumerate(data_info_synced):
            Y = _one_raw_data['y.label']
            seq_id = _one_raw_data['seq_id']
            test_id = "{}_{}".format(seq_id, Y)
            # print("Test ID: {}".format(test_id))
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
            X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
            X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size), axis=-1)
            X = None
            display = None
            X_dense_depth_map_data_display = None
            for _idx_fp, _camera_fp_origin in enumerate(_one_raw_data['x.camera.fps']):
                cnt += 1

                _camera_fp_info = _camera_fp_origin.split('new_college/raw_data')
                _camera_fp = "{}{}".format(config.raw_data.root_dir, _camera_fp_info[-1])
                X_camera_data = cv2.imread(_camera_fp)
                X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                X_camera_data = cv2.resize(X_camera_data, expected_size)
                _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)
                if config.debug_mode:
                    if display is None:
                        X_dense_depth_map_data_display = cv2.applyColorMap(X_dense_depth_map_data.astype(np.uint8),
                                                                           cv2.COLORMAP_JET)
                        display = cv2.addWeighted(X_camera_data, 0.5, X_dense_depth_map_data_display, 0.5, 1)
                    else:
                        display = np.concatenate(
                            [display, cv2.addWeighted(X_camera_data, 0.5, X_dense_depth_map_data_display, 0.5, 1)], 0)
                if X is None:
                    X = _X
                else:
                    X = np.concatenate([X, _X], -1)
            example_dict = {
                config.training_data.features.X.feature_name: X.astype(np.float32),
                config.training_data.features.Y.feature_name: Y,
                "test_id": test_id,
                'drive': 'test'

            }
            yield example_dict


if __name__ == '__main__':
    fire.Fire(NewerCollegeBenchmarksOverGT)
