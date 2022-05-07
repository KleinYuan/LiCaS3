# By default let's not use CUDA for data generation, so that you can do it while running training jobs

create-newer-college-data:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python newer_college_data_generator.py run \
	--config_fp configs/newer_college.yaml \
	--re_sync False

create-kitti-data:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python kitti_data_generator.py run \
	--config_fp configs/kitti.yaml

create-newer-college-labelling-data:
	export CUDA_VISIBLE_DEVICES= && export PYTHONPATH='.' && python newer_college_labelling_data_generator.py run \
	--config_fp configs/newer_college.yaml \
	--re_sync False
