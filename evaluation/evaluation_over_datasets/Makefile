### Below are commands for example-level evaluation

batch-benchmark-kitti-h-models:
	export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python batch_benchmark.py run \
	--benchmark_config_template_fp configs/kitti_benchmark_template.yaml \
	--models_config_fp configs/h_models.yaml \
	--model_type h \
	--dataset_name kitti

# This g models somehow sometimes do not work with tensorflow-gpu 1.14. Using CPU instead
batch-benchmark-kitti-g-models-on-cpu:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python batch_benchmark.py run \
	--benchmark_config_template_fp configs/kitti_benchmark_template.yaml \
	--models_config_fp configs/g_models.yaml \
	--model_type g \
	--dataset_name kitti

batch-benchmark-newer-college-psudo-gt:
	export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python batch_benchmark.py run \
	--benchmark_config_template_fp configs/newer_college_benchmark_template.yaml \
	--models_config_fp configs/h_models.yaml \
	--model_type h \
	--dataset_name newer_college_psudo_gt

batch-benchmark-newer-college-gt:
	export CUDA_VISIBLE_DEVICES=0 && export PYTHONPATH='.' && python batch_benchmark.py run \
	--benchmark_config_template_fp configs/newer_college_benchmark_template.yaml \
	--models_config_fp configs/h_models.yaml \
	--model_type h \
	--dataset_name newer_college_gt
