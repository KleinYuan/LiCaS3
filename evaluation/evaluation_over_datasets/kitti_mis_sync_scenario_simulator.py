import cv2
import fire
import numpy as np
from utils import pb_server
from utils.kitti_loader import load_raw_data
from utils import projection
from utils.metrics import print_stats
from utils.reduce_lidar_lines import reduce_lidar_lines
from box import Box
import yaml
from tqdm import tqdm
import random
from copy import deepcopy


class Calibration:
    P = None
    R = None
    T = None


class KITTIMisSyncScenarioSimulator(object):
    calibration = Calibration()

    def run(self, config_fp, dataset, *args, **kwargs):
        print("Step1: Loading configuration file ...")
        config_master = Box(yaml.load(open(config_fp, 'r').read()))

        print("Step2: Select benchmark dataset ...")
        config = config_master.benchmarks[dataset]

        print("Step3: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step4: Run benchmarks (multiple frame test) ...")
        self.generate_demos(config=config, evaluator=evaluator)

    def batch_run(self, config, *args, **kwargs):
        print("Step1: directly get config ...")
        config = config

        print("Step2: Initialize evaluator ...")
        evaluator = pb_server.PBServer(config)

        print("Step3: Run benchmarks (multiple frame test) ...")
        return self.generate_demos(config=config, evaluator=evaluator)

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
        # TODO (kaiwen) the assumption here is that the calibration shares. If you are testing different drives, you
        # will need to tweak where to load the PRT a bit
        P, R, T = data_info_synced["P"], data_info_synced["R"], data_info_synced["T"]
        self.calibration.P = P
        self.calibration.R = R
        self.calibration.T = T
        for _drive, _drive_info in data_info_synced["paired_fp_seq"].items():
            proposed_end_cam_idxs = []
            simulated_drop_packet = 0
            for _seq, _paired_frame in enumerate(_drive_info):
                if _seq < int(config.training_data.skip_frames):
                    continue
                # cnt = 0
                # (-6, 1) -> (-6, -5, -4, -3, -2, -1, 0)
                _i = random.choice(range(- sampling_window * sampling_stride, 1, sampling_stride))
                cnt = range(- sampling_window * sampling_stride, 1, sampling_stride).index(_i)
                if _seq + _i < 0:
                    continue
                if (_seq + num_frames_in_seq * sampling_stride) >= len(_drive_info):
                    continue
                example_dicts = []
                # Below logics is to avoid backwards offset

                proposed_start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride + (
                        num_frames_in_seq - 1) * sampling_stride
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
                display = None
                X_dense_depth_map_data_display = None
                for _sub_seq in range(0, num_frames_in_seq):
                    start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride + _sub_seq * sampling_stride
                    end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1

                    example = {
                        'x.lidar.fp': _drive_info[_seq + _sub_seq * sampling_stride][0],  # _paired_frame[0],
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
                            config.training_data.reduce_lidar_line_to),
                                                      original_lines=int(config.sensors_info.lidar.beams))
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
                    expected_display_size = (
                        int(config.training_data.features.X.W) * 3, int(config.training_data.features.X.H) * 5)
                    X_dense_depth_map_data_size = (X_dense_depth_map_data.shape[1], X_dense_depth_map_data.shape[0])
                    
                    if len(example['x.camera.fps']) != config.training_data.sampling_window + 1:
                        print("It shall have {} data but only got {} instead.".format(
                            config.training_data.sampling_window + 1, len(example['x.camera.fps'])))
                        continue

                    X = None
                    X_dense_depth_map_data_display = np.expand_dims(
                        cv2.resize(deepcopy(X_dense_depth_map_data), expected_display_size),
                        axis=-1)  # Resize
                    X_dense_depth_map_data = X_dense_depth_map_data[crop_h_start:crop_h_end, :, :]
                    X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size),
                                                            axis=-1)  # Resize

                    for _idx_fp, _camera_fp in enumerate(example['x.camera.fps']):
                        X_camera_data = cv2.imread(_camera_fp)
                        X_camera_data_display = deepcopy(X_camera_data)
                        X_camera_data_display = cv2.resize(X_camera_data_display, expected_display_size)

                        X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                        X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]

                        X_camera_data = cv2.resize(X_camera_data, expected_size)
                        _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)

                        if X_dense_depth_map_data_display.shape[-1] == 1:
                            X_dense_depth_map_data_display = cv2.applyColorMap(
                                X_dense_depth_map_data_display.astype(np.uint8),
                                cv2.COLORMAP_MAGMA)
                        if display is None:
                            display = cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display, 0.8,
                                                      1)
                        else:
                            display = np.concatenate(
                                [display,
                                 cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display, 0.8, 1)],
                                1)
                        if _idx_fp == Y:
                            display_gt = cv2.addWeighted(X_camera_data_display, 0.5, X_dense_depth_map_data_display,
                                                         0.8, 1)

                        if X is None:
                            X = _X
                        else:
                            X = np.concatenate([X, _X], -1)
                    display = np.concatenate([display_gt, display], 1)
                    example_dict = {
                        config.training_data.features.X.feature_name: X.astype(np.float32),
                        config.training_data.features.Y.feature_name: Y,
                        'drive': example['drive'],
                        'seq': example['seq'],
                        'display': display
                    }
                    example_dicts.append(example_dict)
                if len(example_dicts) < num_frames_in_seq:
                    yield None
                yield example_dicts

    @staticmethod
    def write_text(img, text, x, y, fontColor=(0, 0, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 1
        lineType = 2

        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    def generate_demos(self, config, evaluator):
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
        latencies = {

        }
        for inference_data_seq in tqdm(inference_data_generator):
            preds = []
            losses = None
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
                drive_number = inference_data["drive"]
                if model_type == 'g':
                    [pred_synced_frame, losses] = evaluator.inference([X], profile_export=None)
                    pred_synced_frame = pred_synced_frame[0]
                    losses = losses[0]
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
            #  Write example:
            if drive_number not in latencies:
                latencies[drive_number] = {
                    "seq": [int(inference_data_seq[0]["seq"].split(".")[0])],
                    "hardware-sync": [Y_gt],
                    "licas3": [pred]
                }
            else:
                latencies[drive_number]['seq'].append(int(inference_data_seq[0]["seq"].split(".")[0]))
                latencies[drive_number]['hardware-sync'].append(Y_gt)
                latencies[drive_number]['licas3'].append(pred)

            display = inference_data_seq[0]["display"]
            display_h, display_w, _ = display.shape
            display_unit_w = inf_w * 3
            drive_number_int = int(inference_data_seq[0]["drive"].split('_')[-2])
            test_id = "{}{}".format(drive_number_int, inference_data_seq[0]["seq"].split(".")[0])
            pred_display = deepcopy(display[:, display_unit_w * (int(pred) + 1):display_unit_w * (int(pred) + 2), :])
            noise_display = deepcopy(display[:, -display_unit_w:, :])
            paddings_template = np.zeros([int(display_h), display_unit_w, 3])
            gt_display = deepcopy(display[:, :display_unit_w, :])
            display_h, display_w, _ = display.shape

            num_seq = int(display_w / display_unit_w) - 1
            vertical_view = display[:, display_unit_w:display_unit_w * 2, :]
            losses_padding = np.zeros_like(vertical_view)
            losses_padding = self.write_text(losses_padding, "{}".format(losses[0]), 30, 30, fontColor=(255, 255, 255))
            vertical_view_with_losses = np.concatenate([vertical_view, losses_padding], 1)
            for i in range(2, num_seq + 1):
                next_loss_padding = np.zeros_like(losses_padding)
                next_loss_padding = self.write_text(next_loss_padding, "{}".format(losses[i - 1]), 30, 30,
                                                    fontColor=(255, 255, 255))
                next_vertical_view_with_losses = np.concatenate(
                    [display[:, display_unit_w * i:display_unit_w * (i + 1), :], next_loss_padding], 1)
                vertical_view_with_losses = np.concatenate([vertical_view_with_losses, next_vertical_view_with_losses],
                                                           0)
                vertical_view = np.concatenate(
                    [vertical_view, display[:, display_unit_w * i:display_unit_w * (i + 1), :]], 0)
            vertical_view = np.concatenate([vertical_view, pred_display, gt_display], 0)
            cv2.imwrite("simulation_video/kitti/vertical/{}_{}_pred@{}.png".format(test_id, Y_gt, pred), vertical_view)
            cv2.imwrite("simulation_video/kitti/vertical/{}_{}_pred@{}_loss.png".format(test_id, Y_gt, pred),
                        vertical_view_with_losses)

            display = np.concatenate([noise_display, pred_display, gt_display], 1)
            noise_paddings = np.zeros_like(paddings_template)
            licas3_paddings = np.zeros_like(paddings_template)
            gt_paddings = np.zeros_like(paddings_template)
            noise_paddings = self.write_text(noise_paddings, 'No Sync', 30, 30, fontColor=(255, 255, 255))
            licas3_paddings = self.write_text(licas3_paddings, 'LiCaS3', 30, 30, fontColor=(255, 255, 255))
            if int(pred) == Y_gt:
                licas3_paddings = self.write_text(licas3_paddings, 'Pred: {}'.format(pred), display_unit_w - 150, 30,
                                                  fontColor=(0, 255, 0))
            elif abs(int(pred) - Y_gt) < 2:
                licas3_paddings = self.write_text(licas3_paddings, 'Pred: {}'.format(pred), display_unit_w - 150, 30,
                                                  fontColor=(0, 255, 255))
            else:
                licas3_paddings = self.write_text(licas3_paddings, 'Pred: {}'.format(pred), display_unit_w - 150, 30,
                                                  fontColor=(0, 0, 255))
            if (metrics[drive_number]['true'] + metrics[drive_number]['false']) > 0:
                A = metrics[drive_number]['true'] / (metrics[drive_number]['true'] + metrics[drive_number]['false'])
                A2 = (metrics[drive_number]['true'] + metrics[drive_number]['true-neighbour']) / (
                        metrics[drive_number]['true'] + metrics[drive_number]['false'])
                A = np.round(A * 100, 2)
                A2 = np.round(A2 * 100, 2)
                licas3_paddings = self.write_text(licas3_paddings, 'A (A2): {}% ({}%)'.format(A, A2), 30, 70,
                                                  fontColor=(255, 255, 255))
            gt_paddings = self.write_text(gt_paddings, 'Hardware Sync', 30, 30, fontColor=(255, 255, 255))
            gt_paddings = self.write_text(gt_paddings, 'GT: {}'.format(Y_gt), display_unit_w - 150, 30,
                                          fontColor=(255, 255, 255))
            padddings = np.concatenate([noise_paddings, licas3_paddings, gt_paddings], 1)
            display = np.concatenate([display, padddings], 0)

            cv2.imwrite("simulation_video/kitti/{}_{}_pred@{}.png".format(test_id, Y_gt, pred), display)
            # Write example:

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
        print(latencies)

        return metrics


if __name__ == '__main__':
    fire.Fire(KITTIMisSyncScenarioSimulator)
