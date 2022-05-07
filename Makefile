clean:
	docker rm -f $$(docker ps -qa)

clean-swap:
	sudo swapoff -a && sudo swapon -a

# Environment for Training

build:
	docker build -t licas3-docker .

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
        -v /media/kaiwen/extended/licas3_open_source_playground:/root/kitti \
      	licas3-docker bash


# Training commands with LiCaS3
train-licas3-kitti:
	export CUDA_VISIBLE_DEVICES=0 && python commander.py train \
	--model_name "licas3"  \
	--datasets_name "kitti"

train-licas3-newer-college:
	export CUDA_VISIBLE_DEVICES=0 && python commander.py train \
	--model_name "licas3"  \
	--datasets_name "newer_college"

# Training commands with Supervised Learning Model - SL
train-sl-kitti:
	export CUDA_VISIBLE_DEVICES=0 && python commander.py train \
	--model_name "sl"  \
	--datasets_name "kitti"

train-sl-newer-college:
	export CUDA_VISIBLE_DEVICES=0 && python commander.py train \
	--model_name "sl"  \
	--datasets_name "newer_college"