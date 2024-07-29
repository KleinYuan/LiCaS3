# LiCaS3

Here's the official implementation of the Paper:
[K. Yuan, L. Ding, M. Abdelfattah and Z. J. Wang, "LiCaS3: A Simple LiDAR–Camera Self-Supervised Synchronization Method," in IEEE Transactions on Robotics, doi: 10.1109/TRO.2022.3167455.](https://ieeexplore.ieee.org/document/9770125)

![demo-8x mov](https://user-images.githubusercontent.com/8921629/167262935-107c7665-66d5-4ae0-8cef-39dde75451d1.gif)

(speed x 8)

## Project Walk-Thru

This project consists of four folders/parts:

- [X] [training data generation / serialization](training_data_serialization)
- [X] [training](training)
- [X] [evaluation](evaluation)
  - [evaluation over datasets](evaluation/evaluation_over_datasets)
  - [evaluation with limo](evaluation/evaluation_with_limo)
- [X] [complementary newer college datasets](complementary_newer_college_datasets)

Each folder is independent so that you can run them separately. You can find the details of training data generation
and training in this README while the evaluation has its own READMEs, since some evaluation requires quite a lot of setups.
Demonstrative pre-trained weights are provided, and please refer to the [README.md](evaluation/README.md) in evaluation folder.
Raw results of evaluation with limo are also provided for whoever would like to compare my results and plot it on the same 
figure, referring to the [evaluation with limo](evaluation/evaluation_with_limo) sub-folder.


## Hardware Requirements

I conducted my experiments with the following hardware setups, which is highly recommended but not necessarily a hard
requirement.

| Item | Version|
|---|---|
| System | 18.04 (x86), referring to [MACHINE-SETUP.md](https://github.com/KleinYuan/RGGNet/blob/master/MACHINE-SETUP.md) to install the system|
| SSD | 2T |
| GPU Memory | >= 11 GB |

## Main Software Requirements

| Item | Version        |
|---|----------------|
| Python | 3.7.x          |
| NVIDIA-Driver | 440.36         |
| CUDA  | 10.2           |
| cuDNN| v7.4.2         |
| Tensorflow-GPU | >=1.14  &  < 2 |
| Opencv | 4.2            |

I use anaconda to manage virtual environment and except for NVIDIA-Driver and CUDA, all other 
dependencies are mostly used for training data generation. All training happens inside docker
so that the environment is transferable.

## Prepare Raw Datasets


### KITTI

OK, first of all, you shall download the KITTI datasets:

- [X] Navigate to training_data_serialization
- [X] Copy [bash script](training_data_serialization/download.sh) to where you wanna download the data (1-2 TB SSD recommended)
- [X] Run the bash `bash download.sh` to download the `sequence 26_09` to your file dir (and this may take a while, like, a really long while depending on your network speeeeeed)
- [X] Manually split the training and testing datasets, and here's how my folder setup look like (`tree $(YOUR_FOLDER}/kitti/raw/ -L 3`):

The overall size of the raw data is fairly small, 62.3 GB.

```
├── test (8.3 GB)
│   └── 2011_09_26
│       ├── 2011_09_26_drive_0005_sync
│       ├── 2011_09_26_drive_0070_sync
│       ├── calib_cam_to_cam.txt
│       ├── calib_imu_to_velo.txt
│       └── calib_velo_to_cam.txt
└── train (53 GB)
    ├── 2011_09_26
    │   ├── 2011_09_26_drive_0001_sync
    │   ├── 2011_09_26_drive_0002_sync
    │   ├── .......
    │   ├── 2011_09_26_drive_0117_sync
    │   ├── calib_cam_to_cam.txt
    │   ├── calib_imu_to_velo.txt
    │   └── calib_velo_to_cam.txt
    └── download.sh
```

### Newer College

You will need to request the download from this link:https://ori-drs.github.io/newer-college-dataset/

And I organize my folder like below, which is 38.3 GB in total.

```
├── raw_data
│   └── infra1 (45, 880 items, 9.2 GB)
    │   ├── infra1_1583836591_152386717.png
    │   ├── ......
│   └── ouster_scan (15, 302 items, 29.1 GB)
    │   ├── cloud_1583836591_182590976.pcd
    │   ├── ......
│   └── timeoffset
    │   ├── nc-short-time-offsets.csv
```

## Generate/Serialize Training Data

Note: all below happens on your host machine instead of the docker container, mainly due to visualization requirements.

Depending on the stride, window, down-sample ratio, the volume of the datasets will really vary. To be safe, I just 
recommend you get a 1 TB SSD, which is what I used.

### Code Walk-thru

Since we stick to tensorflow 1.x, all training data are created before-hand as a separate process
in tfrecord format.

Below is the tree of the subfolder `training_data_serialization`, which is all you need to create the 
tfrecords:

```
training_data_serialization/
├── configs
│   ├── kitti.yaml
│   └── newer_college.yaml
├── data_generation_base.py
├── download.sh
├── __init__.py
├── kitti_data_generator.py
├── Makefile
├── newer_college_data_generator.py
├── newer_college_labelling_data_genearator.py
└── utils
    ├── __init__.py
    ├── kitti_loader.py
    ├── projection.py
    └── reduce_lidar_lines.py
```


It simply contains 3 main parts: generator scripts, config yamls and utils. [utils](training_data_serialization/utils) is
simply a folder containing helper functions of projections, data loader or reduce lidar lines. We only have two configs for
kitti and newer college respectively. Beside training data configs, a redundant sensor information is also included for 
references. Finally, we have two data generator, which all are inherited from the 
[Generator class in data_generator_base.py](training_data_serialization/data_generator_base.py). As you can see 
from the class, regardless of the logics, the main flow is just three steps: load config, generate training data, serialize
into tfrecords on disk. I also provided the script ([newer_college_labelling_data_generator.py](training_data_serialization/newer_college_labelling_data_genearator.py)))
that I used to create the annotation datasets (which I used for benchmark only). You pbbly don't need this but I still 
put it here for references.


### KITTI

First, take a look at the [kitti.yaml](training_data_serialization/configs/kitti.yaml):

Below are all fields that you will need to update before creating the tfrecords:

```
raw_data:
  ...
  root_dir: "Replace this with your training / validation / testing raw data sub-folder"

training_data:
  ...
  output_dir: "Replace this with where you would like to save the tfrecords"
  name:  "training" #  ["training", "testing", "validation"]
  sampling_window: 5 # by default, we use a single lidar frame
  sampling_stride: 1 # skip every every two frames
  features:
    X:
      ...
      C: 24 # (sampling_window + 1) * 4

```

It shall be noted that `sampling_window` here corresponds to `l-1` in the paper. This is because computer
starts from 0.

The comments shall explain itself very well. After this is done, just with one command as below, you shall
be able to see the data generation running on multi-processors:

```
# navigate to the subfolder training_data_serialization
make create-kitti-data
```

Unfortunately, you will need to run training/validation/testing data generation separately. However, you can simply
open three terminals and update the yaml on-the-fly to do the job. 

You can tweak the following based on your compute power.

```
training_data:
  chunk_size: 5000
  downsample_ratio: 0.5 # TODO: This depends on how big the machine is
```

It will take a while to finish the training data generation. Taking sample window as 5, sampling stride as 1, 
downsample_ratio as 0.1 as an example, on my Intel Core i9-9900K CPU (16 cores, 32 GB Memory), it will take around 15 mins
to serialize a 6.4 GB (2 files in total) training data. 

### Newer College

Like-wise, you will need to update similar fields in the [newer_college.yaml](training_data_serialization/configs/newer_college.yaml):

```
raw_data:
  ...
  root_dir: "Replace this with your raw data folder"
  ...
  time_offsets_csv_fp: "Replace this with your timeoffset csv path" # timeoffset/nc-short-time-offsets.csv
  generated_fp:
    synced_raw_data_info: "Replace this with where you would like to save the synced raw data info" # xxxx/synced_raw_data_info.pkl
training_data:
  output_dir: "Replace this with where you would like to save the tfrecords"
  sampling_stride: 1
  sampling_window: 5
  features:
    X:
      ...
      C: 24  # (sampling_window + 1) * 4
```

The main difference is that you don't have to run the scripts multiple times manually for training/validation/testing since
the split ratio does the trick. Note: the `time_offsets_csv_fp` is only used for metrics and the baseline (not really
ground truth since it's quite poor) offset is not used as supervising signal.

```
# navigate to the subfolder training_data_serialization
make create-newer-college-data
```

## Training

Note: all below happens inside the docker container, so that you can pbbly run it everywhere.

### Build the Image and Enter a Shell

Build the docker image: 

```
make build
```

Update the volume mount of your training data in the [Makefile](Makefile):

```
run:
	sudo docker run -it \
        --runtime=nvidia \
        --name="licas3-experiment" \
        --net=host \
        --privileged=true \
        --ipc=host \
        --memory="10g" \
        --memory-swap="10g" \
        -v ${PWD}:/root/licas3 \
        -v ${PWD}/newer_college:/root/newer_college \
        -v ${PWD}/kitti:/root/kitti \
      	licas3-docker bash
```

I by default believe that you put newer_college and kitti data under `${PWD}`. However, if that's not true, update it.

At last, enter a shell with simply doing the following:

```
make run
```

And then you shall enter a shell with exact same environment of what I was using!


### Experimental Data Management Suggestions

In you are working on a paper based on my works, which also needs to do ablation studies with various stride/window settings,
I highly recommend you organize your training/validation/testing datasets as follows (taking kitti as an example):

```
kitti
├── raw
├── training_datasets
│   ├── sample_01_stride_05_beam_64_downsample_05/
│                   ├── training
│                         ├── 0_e496049da47a41ea9ebaabdbc9a2ee6f.tfrecord
│                         ├── 1_e5fc9b31f1864279b5e2bc5ddc241347.tfrecord
│                         ├── .....tfrecord
│                   ├── validation
│                   ├── testing # technically you shouldn't include this in the training process but benchmark
│   └── sample_02_stride_05_beam_64_downsample_05/
│   └── ..../
```

### Run Training

#### Train LiCaS3 against KITTI

Firstly, you shall navigate to the [training/configs](training/configs) folder 
and update the following fields:

```
name: 'licas3_kitti_sample_01_stride_05_beam_64_downsample_05' # 0. use a good name to make your experiment life easier

tensors:
  placeholders:
    X:
      shape: [32, 256, 24]  # 1. make sure this is consistent with your training data

data:
  num_parallel_reads: 16
  inputs:
    X:
      C: 24   # 2. make sure this is consistent with your training data and same as your placeholders
  tfrecords_train_dirs:
    - "/root/kitti/training_datasets/sample_01_stride_05_beam_64_downsample_05/training" # 3. training data location

  tfrecords_test_dirs:  # Test in training means "validation", which you shouldn't use the real testing datasets
    - "/root/kitti/training_datasets/sample_01_stride_05_beam_64_downsample_05/validation" # 4. validation data location

inference:
  included_tensor_names:
  - 'licas3_kitti_sample_01_stride_05_beam_64_downsample_05'  # 5*. this is only used when you try to create inference graph

  freeze:
    output_node_name: 'licas3_kitti_sample_01_stride_05_beam_64_downsample_05/prediction_from_classifier' # 6*. this is only used when you try to create inference graph
```

After you have the correct configs, and inside the shell, you can simply do 

```
make train-licas3-kitti
```

which is equal to 

```
export CUDA_VISIBLE_DEVICES=0 && python commander.py train \
	--model_name "licas3"  \
	--datasets_name "kitti"
```

And then you shall see your training starts and terminal printing logs:

```
FO:tensorflow:  [Stage 1 - 1 th iteration] Train loss: 0.37371593713760376
INFO:tensorflow:  [Stage 1 - 2 th iteration] Train loss: 0.39040398597717285
INFO:tensorflow:  [Stage 1 - 3 th iteration] Train loss: 0.38538438081741333
....
```

The checkpoints will be saved to [results/save/](results/save) and log will be saved to [results/log/](results/log). You
can use a separate terminal to launch tensorboard to check the training!

It shall be noted that to better manage the experiments, I also append a unix-epoch timestamp at the end of the experiment
name. So you will have everything saved as something like `licas3_kitti_sample_01_stride_05_beam_64_downsample_05_16xxxxx.xxx`.

#### Train LiCaS3 against Newer College

This is nothing too different from KITTI training. The [config](training/configs/licas3_newer_college.yaml) 
is the only thing you need to update.

Once it's done, inside the shell, simply do 

```
make train-licas3-newer-college
```

#### Train Supervised Learning Models

Same thing, same thing! Update the [configs/sl_newer_college.yaml](training/configs/sl_newer_college.yaml)  or 
[configs/sl_kitti.yaml](training/configs/sl_kitti.yaml). Then, inside the shell, simply do 

```
make train-sl-newer-college
```

or 

```
make train-sl-kitti
```


## Evaluations

Please check the evaluation folder [README.md](evaluation/README.md).


## Citation

https://ieeexplore.ieee.org/document/9770125

```
@ARTICLE{9770125,
  author={Yuan, Kaiwen and Ding, Li and Abdelfattah, Mazen and Wang, Z. Jane},
  journal={IEEE Transactions on Robotics}, 
  title={LiCaS3: A Simple LiDAR–Camera Self-Supervised Synchronization Method}, 
  year={2022},
  volume={38},
  number={5},
  pages={3203-3218},
  doi={10.1109/TRO.2022.3167455}}
```


## Clarification

This repo is a largely refactored open-source version based on my internal experimental repository (which is really messy) for my publications.
If you see potential issues/bugs or have questions regarding my works, please feel free to email me (kaiwen dot yuan1992 at gmail dot com). As I graduated, UBC widthdrew my school email kaiwen@ece.ubc.ca, which is not valid any more.

This code is largely based (not dependent but borrowed many codes) on my previous projects: [RGGNet](https://github.com/KleinYuan/RGGNet) and
 [tf-blocks](https://github.com/KleinYuan/tf-blocks), which further depends on tensorflow 1.x. Every software has a limited life. If one day you find this project cannot be pulled-and-run like 2022, it's
not the end of the world. The idea of this work is super simple, and here's the one-liner: `Train a mono depth estimator with mis-synched pairs within latency bound,
to generate self-labels to train another classifier for integral offset estimation, all in a self-supervised manner`.
I believe that who are interested can implement it in a short time.

If you are interested in collaborations with me on related topics, don't hesitate to reach out to me :)

