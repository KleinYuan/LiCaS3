# Evaluation

In our works, we have evaluated the methods from various levels:

- example level: treating it as an ML task, compute metrics of learning task
- sequence level: simulate a real mis-synched scenario and evaluate how much it has been corrected
- end-to-end level: integrated with off-the-shelf SLAM algorithm, [LIMO](https://ieeexplore.ieee.org/abstract/document/8594394/)

This readme walks thru the preparations you need to do before you can run the evaluations. At the
end, I forward you to the subfolder's documentation for more detailed steps.

## Preparations

(This section only applies to those who do the training and would like to continue.)

As you can see, there are many operators in the [licas3_model](../training/models/licas3_model.py) that are useless for 
evaluation. Especially if you would like to deploy this model, your inference graph is pbbly not even the same as the training
graph. Namely, the first step is to load the pre-trained model weights into an inference graph, then freeze it. This can
even make your evaluations faster! This is also how we obtained the evaluation protobufs from checkpoints.

Go to [training/experiments](../training/experiments) folder, and you will see the following three modules:

- [licas3_inference_h.py](../training/experiments/licas3_inference_h.py)
- [licas3_inference_g.py](../training/experiments/licas3_inference_g.py)
- [sl_inference_h.py](../training/experiments/sl_inference_h.py)

which corresponds to the `h` and `g` model (supervised learning model only has `h` model) of each method.

### Create Inference Protobuf

Take kitti as an example. Assume your trained checkpoints are saved under 
`results/save/licas3_kitti_sample_01_stride_05_beam_64_downsample_05/1619474131.9902055`, then you can run
the following inside the docker:

```
python -m training.training.licas3_inference_h process \
--config_fp ../configs/licas3_kitti.yaml \
--from_dir results/save/licas3_kitti_sample_01_stride_05_beam_64_downsample_05/1619474131.9902055 \ # Update this
--to_dir inference/licas3_kitti_sample_01_stride_05_beam_64_downsample05@1619474131.9902055 \
--to_name best
```

(It's important that you append the timestamp after the experiment name separated by `@`.)

Then you shall see a frozen inference h model from your licas3 kitti experiment saved under `inference` folder.

Similarly, you can do the same to the rest with the correct configurations:


```
python -m training.training.licas3_inference_g process \
--config_fp ../configs/licas3_kitti.yaml \
--from_dir results/${UPDATE_THIS \ 
--to_dir inference/${UPDATE_THIS} \
--to_name best
```

```
python -m training.training.sl_inference_h process \
--config_fp ../configs/sl_kitti.yaml \
--from_dir results/${UPDATE_THIS \ 
--to_dir inference/${UPDATE_THIS} \
--to_name best
```


One potential issue that you may meet in future is that the frozen g model may have some issue while doing inference at
the same environment with ROS. If you meet this issue, you can do the following instead to force all operator conversion
happen in CPU instead of GPU. Except for your later benchmark will be come a litttttttle bit slower, no hurts at all!

```
export CUDA_VISIBLE_DEVICES= && python -m training.training.licas3_inference_g process \
--config_fp ../configs/licas3_kitti.yaml \
--from_dir results/${UPDATE_THIS \ 
--to_dir inference/${UPDATE_THIS} \
--to_name best
```

### Download Extra Newer College Labelling

As described in the paper that we have manually labelled a small datasets of Newer College, you may need this 
during the evaluation process. You can skip this if you are not interested in comparing LiCaS3 with the baseline.

Check [complementary_newer_college_datasets/README.md](../complementary_newer_college_datasets/README.md) for more details.


### Pre-trained Model

Pre-trained models can be downloaded with this link: 
https://drive.google.com/drive/folders/15Cn-jAQCT99uGGLXfIeq_m9l1o0Efkl9?usp=sharing


### Run Evaluations

For the three levels evaluations, we split them into two folders since the example-level and scenario-level
shares lots of code and all test over datasets, instead of ROS:

- [X] Evaluation over datasets 
  - (example-level) Benchmark over testing examples: check [evaluation_over_datasets/README.md](evaluation_over_datasets/README.md)
  - (scenario-level) Evaluation over simulated scenario with testing raw data: check [evaluation_over_datasets/README.md](evaluation_over_datasets/README.md)
- [X] (end-to-end-level) Evaluation over off-the-shelf SLAM algorithm, LIMO: check [evaluation_with_limo/README.md](evaluation_with_limo/README.md)


