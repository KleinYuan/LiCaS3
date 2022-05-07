import cv2
import fire
import numpy as np
from utils.kitti_loader import load_raw_data
from utils import pb_server, projection
from utils.reduce_lidar_lines import reduce_lidar_lines
from utils.metrics import print_stats
from box import Box
import yaml
from tqdm import tqdm
import random


class Calibration:
    P = None
    R = None
    T = None


class KITTIBenchmarks(object):
    """
    ├── kitti
        ├── synced_and_rectified
            ├── 2011_09_26
                ├── 2011_09_26_drive_0001_sync
                ├── 2011_09_26_drive_0002_sync
                ....
                ├── 2011_09_26_drive_0117_sync
                ├── calib_cam_to_cam.txt
                ├── calib_imu_to_velo.txt
                ├── calib_velo_to_cam.txt
    """
    calibration = Calibration()

    def run(self, config_fp, dataset, *args, **kwargs):
        print("Step1: Loading configuration file ...")
        config_master = Box(yaml.load(open(config_fp, 'r').read()))

        print("Step2: Select benchmark dataset ...")
        config = config_master.benchmarks[dataset]

        print("Step3: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step4: Run benchmarks (multiple frame test) ...")
        self.run_benchmarks_multi_frame_test(config=config, evaluator=evaluator)

    def batch_run(self, config, *args, **kwargs):
        print("Step1: directly get config ...")
        config = config

        print("Step2: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step3: Run benchmarks (multiple frame test) ...")
        return self.run_benchmarks_multi_frame_test(config=config, evaluator=evaluator)


    def generate_inference_data_multi_frame(self, config, *args, **kwargs):
        num_frames_in_seq = int(config.training_data.multi_frame_test.num_frames_in_seq)
        datasets_name = config.name
        crop_h_start = int(config.training_data.crop_shape[0][0])
        crop_h_end = int(config.training_data.crop_shape[0][1])

        camera_sensor_H, camera_sensor_W = config.sensors_info.camera.shape
        # KITTI is already hardware synced
        data_info_synced = load_raw_data(data_fp=config.raw_data.root_dir)
        print("Synced example number is : {}".format(len(data_info_synced["paired_fp_seq"])))
        sampling_window = config.training_data.sampling_window
        sampling_stride = config.training_data.sampling_stride
        print("Sliding window is {}".format(sampling_window))  # 6
        P, R, T = data_info_synced["P"], data_info_synced["R"], data_info_synced["T"]
        self.calibration.P = P
        self.calibration.R = R
        self.calibration.T = T
        for _drive, _drive_info in data_info_synced["paired_fp_seq"].items():
            for _seq, _paired_frame in enumerate(_drive_info):
                if _seq < int(config.training_data.skip_frames):
                    continue
                # cnt = 0
                # (-6, 1) -> (-6, -5, -4, -3, -2, -1, 0)
                _i = random.choice(range(- sampling_window * sampling_stride, 1, sampling_stride))
                #for _i in range(- sampling_window * sampling_stride, 1, sampling_stride):
                cnt = range(- sampling_window * sampling_stride, 1, sampling_stride).index(_i)
                if _seq + _i < 0:
                    continue
                if (_seq + num_frames_in_seq * sampling_stride) >= len(_drive_info):
                    continue
                example_dicts = []
                label_rel_idx = sampling_window - cnt
                for _sub_seq in range(0, num_frames_in_seq):
                    start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride + _sub_seq * sampling_stride
                    end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1

                    example = {
                        'x.lidar.fp': _drive_info[_seq+_sub_seq * sampling_stride][0], # _paired_frame[0],
                        'x.camera.fps': [x[1] for x in _drive_info[start_cam_idx: end_cam_idx: sampling_stride]],
                        'y.label': label_rel_idx,
                        'drive': _paired_frame[0].split('/')[-4],
                        'seq_id': _seq,
                        'seq': _paired_frame[0].split('/')[-1].split('.png')[0],
                    }
                    Y = example['y.label']
                    P, R, T = self.calibration.P, self.calibration.R, self.calibration.T
                    pts_xyzi = np.fromfile(example['x.lidar.fp'], dtype=np.float32).reshape((-1, 4))
                    if 'reduce_lidar_line_to' in config.training_data:
                        pts_xyzi = reduce_lidar_lines(pts_xyzi, reduce_lidar_line_to=int(
                            config.training_data.reduce_lidar_line_to), original_lines=int(config.sensors_info.lidar.beams))
                    pts_xyz = pts_xyzi[:, :3]

                    X_dense_depth_map_data = projection.get_dense_depth_map(
                        pts_xyz=pts_xyz,
                        H=camera_sensor_H,
                        W=camera_sensor_W,
                        T=T,
                        R=R,
                        P=P,
                        datasets_name=datasets_name,
                        kernel_size=8,
                        norm_methods=config.training_data.z_norm_methods,
                        lidar_range=config.sensors_info.lidar.range
                    )
                    expected_size = (int(config.training_data.features.X.W), int(config.training_data.features.X.H))
                    X_dense_depth_map_data_size = (X_dense_depth_map_data.shape[1], X_dense_depth_map_data.shape[0])
                    
                    if len(example['x.camera.fps']) != config.training_data.sampling_window + 1:
                        print("It shall have {} data but only got {} instead.".format(
                            config.training_data.sampling_window + 1, len(example['x.camera.fps'])))
                        continue

                    X = None
                    X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
                    X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size),
                                                            axis=-1)  # Resize
                    for _idx_fp, _camera_fp in enumerate(example['x.camera.fps']):
                        X_camera_data = cv2.imread(_camera_fp)
                        X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                        X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                        X_camera_data = cv2.resize(X_camera_data, expected_size)
                        _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)

                        if X is None:
                            X = _X
                        else:
                            X = np.concatenate([X, _X], -1)
                    example_dict = {
                        config.training_data.features.X.feature_name: X.astype(np.float32),
                        config.training_data.features.Y.feature_name: Y,
                        'drive': example['drive'],
                        'seq': example['seq']
                    }
                    example_dicts.append(example_dict)
                if len(example_dicts) < num_frames_in_seq:
                    yield None
                yield example_dicts

    def run_benchmarks_multi_frame_test(self, config, evaluator):
        model_type = config.model_type
        print("====== {} model demo generator  ====".format(model_type))
        print("     Step1: Generate inference data ...")
        inference_data_generator = self.generate_inference_data_multi_frame(config=config)
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
                X = [inference_data["X"] / 255.]
                Y = inference_data["Y"]
                if model_type == 'g':
                    [pred_synced_frame, losses] = evaluator.inference([X], profile_export=None)
                    pred_synced_frame = pred_synced_frame[0]  # TODO: heuristic for model g
                    print("\n------\npred_synced_frame: {} with losses {}".format(pred_synced_frame, losses))
                else:
                    [pred_synced_frame] = evaluator.inference([X], profile_export=None)
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


if __name__ == '__main__':
    fire.Fire(KITTIBenchmarks)
