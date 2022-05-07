# Evaluation over Datasets

## Benchmark over Testing Examples: Example Level

Now, I will just assume you have the corresponding frozen inference model created under the folder `inference`. To this 
point, you are pretty much done with the [training](../../training) folder. You will find some scripts under [utils](../evaluation_over_datasets/utils)
are copied over from [../training_data_serialization/utils](../../training_data_serialization/utils). This is done on purpose
so that this evaluation folder is less dependent on other folders and future researchers can move it anywhere.

Let's move our eyes on the 
[evaluation/evaluation_over_datasets](../evaluation_over_datasets) folder, which has three runners:

- [X] [kitti_runner.py](../evaluation_over_datasets/kitti_runner.py)
- [X] [newer_college_gt_runner.py](../evaluation_over_datasets/newer_college_gt_runner.py)
- [X] [newer_college_psudo_gt_runner.py](../evaluation_over_datasets/newer_college_psudo_gt_runner.py)

And one script called [batch_benchmark.py](../evaluation_over_datasets/batch_benchmark.py). As it is named, you in fact 
just need to use this [batch_benchmark.py](../evaluation_over_datasets/batch_benchmark.py) to finish all the benchmarks, regardless
of model or datasets.

As long as you have the frozen model prepared as I discussed in the above section, the only action items are to update the configs:

### KITTI h Models

1. Update the directory where you saved your testing tfrecords in [kitti_benchmark_template.yaml](../evaluation_over_datasets/configs/kitti_benchmark_template.yaml) 
```
benchmarks:
  raw_data:
    root_dir: "Replace this with where you saved the testing tfrecords"
```

2. Update the directory where you saved your frozen pbs in [h_models.yaml](../evaluation_over_datasets/configs/h_models.yaml) 

```
models:
  variables:
    num_frames_in_seq: [1, 10, 20] # You can also update this with how many frames you would like to test
  kitti:
    id_h001:
      model_fp: "Replace this with where you saved your pb file"
      l: 5
      s: 1
```

It shall be noted that `l` here corresponds to `l-1` in the paper. This is because computer
starts from 0 but in the paper, I start from 1. I am sorry for this confusion.

I have put some examples in [h_models.yaml](../evaluation_over_datasets/configs/h_models.yaml). Unless you happen to name/save 
everything exactly the same as me, please update them. Since during my experiments, there are 24 h models and 6 g models, 
each of which need to run single frame, 10 and 20 frames. These add up to around 100 benchmarks. Namely, I would really
recommend you give each model a unique id as I did in the above. For example, you can name all licas3 models as `id_h0xx`, 
and all sl models as `id_h1xx`. This can make your later life easier.

(Important) I have a hard coded logic in the batch runner to parse the model name space:

```
model_namespace = model_fp.split('/')[-2].split('@')[0]
```
The hidden assumption here is that your model is named as `${model_name}@${time-stamp}`. If you follow my
previous steps of model freezing, you shouldn't have any problem. However, if 
your model path naming convention is different, please update this line of code.

After you are done with the configurations, simple do:

```
make batch-benchmark-kitti-h-models
```

which is equal to 

```
export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python batch_benchmark.py run \
--benchmark_config_template_fp configs/kitti_benchmark_template.yaml \
--models_config_fp configs/h_models.yaml \
--model_type h \
--dataset_name kitti
```

How it works is that [batch_benchmark.py](../evaluation_over_datasets/batch_benchmark.py) will use the `$benchmark_config_template_fp`
as the template to load `s`, `l` and `model_fp` in the `$models_config_fp`, under `$dataset_name` section. The `$model_type`
is to help the scripts to update the tensor field for different graphs.

### Benchmark the Rest

Follow the exact same procedures above and you can do 

- batch-benchmark-kitti-g-models-on-cpu: `make batch-benchmark-kitti-g-models-on-cpu`
- batch-benchmark-newer-college-psudo-gt: `batch-benchmark-newer-college-psudo-gt`
- batch-benchmark-newer-college-gt: `batch-benchmark-newer-college-gt`

### Check the Results

For each benchmark, a report will be saved on wherever you configured. The report looks like this:

```
{
    'total': 
        {
            'true': 10391, 
            'false': 2913, 
            'true-neighbour': 2638
        }, 
    None: 
        {
            'true': 10391, 
            'false': 2913, 
            'true-neighbour': 2638
        }
}
```

The `total` and `None` field are identical. `true: 10391` means 10391 frames are correctly predicted while `'false': 2913`
means 2913 frames are wrong (no matter how many frames diff). `'true-neighbour': 2638` means that during the 2913 wrong 
ones, 2638 predictions are only one frame away. Namely, you will get:

- N = 2913 + 10391 (Total examples)
- N_t = 10391
- N_d1 = 2638
- A = N_t / N = 78.10%
- A_{t2} = (N_t + N_d1)  / N = 97.93%

which corresponds to 4-th row in Table V. 

## Evaluation over Simulated Scenario with Testing Raw Data: Scenario Level

The above section reports the performance of the ML task. Now, we talk about evaluations of LiCaS3
for a simulated real mis-synchronized scenario.

For both datasets, the process is very similar to the example level except that we create mis-sync
scenario from the raw data on the fly.


For KITTI datasets, run the following with your `$simulation_model_id`:

```
export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python mis_sync_scenario_simulator.py run \
--benchmark_config_template_fp configs/kitti_benchmark_template.yaml \
--models_config_fp configs/h_models.yaml \
--model_type h \
--dataset_name kitti \
--simulation_model_id id_h003 \
--simulation_num_frames_in_seq 1
```


For Newer College datasets, run the following with your `$simulation_model_id`:

```
export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python mis_sync_scenario_simulator.py run \
--benchmark_config_template_fp configs/newer_college_benchmark_template.yaml \
--models_config_fp configs/h_models.yaml \
--model_type h \
--dataset_name newer_college_psudo_gt \
--simulation_model_id id_h011 \
--simulation_num_frames_in_seq 1
```


Then you shall actually be able to see images that are created! I also provide a tool to convert
the folder of images into a video: [folder2video.py](folder2video.py).


(Important) Same as the previous section, I have a hard coded logic in the simulation runner
to parse the model name space:

```
model_namespace = model_fp.split('/')[-2].split('@')[0]
```

The hidden assumption here is that your model is named as `${model_name}@${time-stamp}`. If you follow my
previous steps of model freezing, you shouldn't have any problem. However, if 
your model path naming convention is different, please update this line of code.