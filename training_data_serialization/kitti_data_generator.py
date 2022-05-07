import cv2
import fire
from data_generator_base import Generator
import multiprocessing
import numpy as np
import os
import pathlib
import random
import tensorflow as tf
from tqdm import tqdm
from utils.kitti_loader import load_raw_data
from utils import projection
from utils.reduce_lidar_lines import reduce_lidar_lines
import uuid


class Calibration:
    P = None
    R = None
    T = None

class KITTIGenerator(Generator):
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
    def generate_training_data(self, config, visualization=False, *args, **kwargs):
        """
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
        training_examples_data_info = [
            [example1, example2, ....], # belongs to one segments
            [],
            []
        ]
        """
        datasets_name = config.name
        # KITTI is already hardware synced
        data_info_synced = load_raw_data(data_fp=config.raw_data.root_dir)

        sampling_window = config.training_data.sampling_window
        sampling_stride = config.training_data.sampling_stride
        print("Sliding window is {}".format(sampling_window))  # 6
        training_examples_data_info = []
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
                #_i = random.choice(range(- sampling_window * sampling_stride, 1, sampling_stride))
                for _i in range(- sampling_window * sampling_stride, 1, sampling_stride):
                    cnt = range(- sampling_window * sampling_stride, 1, sampling_stride).index(_i)
                    if _seq + _i < 0:
                        continue
                    # TODO: consider moving those hard-coded keys to config
                    start_cam_idx = _seq - (sampling_window - cnt) * sampling_stride
                    end_cam_idx = start_cam_idx + sampling_window * sampling_stride + 1
                    label_rel_idx = sampling_window - cnt
                    example = {
                        'x.lidar.fp': _paired_frame[0],
                        'y.camera.fp': _paired_frame[1],
                        'x.camera.fps': [x[1] for x in _drive_info[start_cam_idx: end_cam_idx: sampling_stride]],
                        'y.label': label_rel_idx,
                        'seq_id': _seq
                    }

                    if visualization:
                        pts_xyz = np.fromfile(example["x.lidar.fp"], dtype=np.float32).reshape((-1, 4))[:, :3]
                        overlay_gt = projection.display_projected_img(pts_xyz, example["y.camera.fp"], T, R, P=P, datasets_name=datasets_name, dense=True)
                        for _x_camera_fp in example["x.camera.fps"]:
                            overlay_offset = projection.display_projected_img(pts_xyz, _x_camera_fp, T, R, P=P, datasets_name=datasets_name, dense=True)
                            overlay = np.concatenate([overlay_offset, overlay_gt], 0)
                            # overlay = cv2.resize(overlay, (208*4, 2*106))
                            cv2.imshow("overlay_gt", overlay)
                            cv2.waitKey(0)

                    training_examples_data_info.append(example)
                # cnt += 1
            print("Push {} examples into the training data".format(len(training_examples_data_info)))
        if visualization:
            cv2.destroyAllWindows()
        print("Generated {} training segments".format(len(training_examples_data_info)))
        # Generated 110215 training examples
        return training_examples_data_info

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def write_text(img, text, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    def generate_training_paris_and_serialize_one_chunk_to_tfrecords(self, config, output_fp, chunk_idx, data_chunk,
                                                                     camera_sensor_H, camera_sensor_W):
        if not os.path.isdir(output_fp):
            print("{} does not exits, creating one.".format(output_fp))
            pathlib.Path(output_fp).mkdir(parents=True, exist_ok=True)
        print("Creating {} th chunk of the tfrecords ...".format(chunk_idx))
        if not config.debug_mode:
            writer = tf.python_io.TFRecordWriter("{}/{}_{}.tfrecord".format(output_fp, chunk_idx, uuid.uuid4().hex))
        datasets_name = config.name
        crop_h_start = int(config.training_data.crop_shape[0][0])
        crop_h_end = int(config.training_data.crop_shape[0][1])
        crop_w_start = int(config.training_data.crop_shape[1][0])
        crop_w_end = int(config.training_data.crop_shape[1][1])

        for _idx, _one_raw_data in enumerate(data_chunk):
            example_dict = {}
            label = _one_raw_data['y.label']
            P, R, T = self.calibration.P, self.calibration.R, self.calibration.T
            pts_xyzi = np.fromfile(_one_raw_data['x.lidar.fp'], dtype=np.float32).reshape((-1, 4))
            if 'reduce_lidar_line_to' in config.training_data:
                pts_xyzi = reduce_lidar_lines(pts_xyzi, reduce_lidar_line_to=int(config.training_data.reduce_lidar_line_to))
            pts_xyz = pts_xyzi[:, :3]
            X_dense_depth_map_data_o = projection.get_dense_depth_map(
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
            cnt = 0
            expected_size = (int(config.training_data.features.X.W), int(config.training_data.features.X.H))
            X_dense_depth_map_data_size = (X_dense_depth_map_data_o.shape[1], X_dense_depth_map_data_o.shape[0])
            
            if len(_one_raw_data['x.camera.fps']) != config.training_data.sampling_window + 1:
                print("It shall have {} data but only got {} instead.".format(
                    config.training_data.sampling_window + 1, len(_one_raw_data['x.camera.fps'])))
                continue

            X = None
            display = None
            X_dense_depth_map_data_display = None
            X_dense_depth_map_data = X_dense_depth_map_data_o[crop_h_start:crop_h_end, :, :]
            X_dense_depth_map_data = np.expand_dims(cv2.resize(X_dense_depth_map_data, expected_size),
                                                    axis=-1)  # Resize
            for _idx_fp, _camera_fp in enumerate(_one_raw_data['x.camera.fps']):
                cnt += 1
                X_camera_data = cv2.imread(_camera_fp)
                if config.debug_mode:
                    cv2.imwrite("depth_map_{}.png".format(cnt),
                                cv2.applyColorMap(X_dense_depth_map_data_o.astype(np.uint8), cv2.COLORMAP_MAGMA))
                    cv2.imwrite("X_camera_data_{}.png".format(cnt), X_camera_data)
                X_camera_data = cv2.resize(X_camera_data, X_dense_depth_map_data_size)
                X_camera_data = X_camera_data[crop_h_start:crop_h_end, :, :]
                X_camera_data = cv2.resize(X_camera_data, expected_size)
                # print(X_camera_data.shape)
                _X = np.concatenate([X_camera_data, X_dense_depth_map_data], -1).astype(np.float32)
                if config.debug_mode:
                    if display is None:
                        X_dense_depth_map_data_display = cv2.applyColorMap(X_dense_depth_map_data.astype(np.uint8),cv2.COLORMAP_MAGMA)

                        display = cv2.addWeighted(X_camera_data, 0.0, X_dense_depth_map_data_display, 0.5, 1)

                    else:
                        display = np.concatenate(
                            [display, cv2.addWeighted(X_camera_data, 0.0, X_dense_depth_map_data_display, 0.5, 1)], 0)

                if X is None:
                    X = _X
                else:
                    X = np.concatenate([X, _X], -1)
            example_dict.update({
                config.training_data.features.X.feature_name: tf.train.Feature(
                    float_list=tf.train.FloatList(value=X.flatten().astype(np.float32))),
                config.training_data.features.Y.feature_name: tf.train.Feature(float_list=tf.train.FloatList(
                    value=[label])),
            })
            example = tf.train.Example(features=tf.train.Features(feature=example_dict))
            if not config.debug_mode:
                writer.write(example.SerializeToString())
            else:
                display= self.write_text(display, "{}".format(label), 30, 30)
                cv2.imshow("kitti.png", display.astype(np.uint8))
                cv2.waitKey(0)
        if not config.debug_mode:
            writer.close()
            print("Created {}.{}.tfrecord".format(output_fp, chunk_idx))
        else:
            cv2.destroyAllWindows()

    def generate_training_paris_and_serialize_to_tfrecords(self, datasets, config, name):
        _data_chunks = self.chunks(lst=datasets, n=config.training_data.chunk_size)
        output_fp = "{}/{}".format(config.training_data.output_dir, name)
        print("Creating {} tfrecord with each including {} examples............".format(
            len(datasets) / config.training_data.chunk_size, config.training_data.chunk_size))

        camera_sensor_H, camera_sensor_W = config.sensors_info.camera.shape
        jobs = []
        for _chunk_idx, _data_chunk in tqdm(enumerate(_data_chunks)):
            if config.debug_mode:
                self.generate_training_paris_and_serialize_one_chunk_to_tfrecords(config, output_fp, _chunk_idx,
                                                                                  _data_chunk, camera_sensor_H,
                                                                                  camera_sensor_W)
            else:
                p = multiprocessing.Process(
                    target=self.generate_training_paris_and_serialize_one_chunk_to_tfrecords,
                    args=(config, output_fp, _chunk_idx, _data_chunk, camera_sensor_H, camera_sensor_W))
                jobs.append(p)
                p.start()

    def serialize_data_into_tfrecords(self, config, training_examples_data_info, *args, **kwargs):
        downsample_to = int(float(config.training_data.downsample_ratio) * len(training_examples_data_info))
        print("Down sampling {} examples to {}".format(len(training_examples_data_info), downsample_to))
        training_examples_data_info = random.sample(training_examples_data_info, downsample_to)
        self.generate_training_paris_and_serialize_to_tfrecords(training_examples_data_info, config, config.training_data.name)

if __name__ == '__main__':
    fire.Fire(KITTIGenerator)
