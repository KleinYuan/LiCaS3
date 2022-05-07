import fire
import subprocess


class Commander(object):

    @staticmethod
    def _create_starter(model_name, datasets_name=None):
        print("Creating starter according to model name ....")
        if datasets_name is None:
            command = '''sed "s|STUB_MODEL_NAME|{}|g" training/train_start_template.py>training/train_start_{}.py'''.format(model_name, model_name)
        else:
            command = '''sed "s|STUB_MODEL_NAME|{}|g; s|STUB_DATASET_NAME|{}|g" training/train_start_template.py>training/train_start_{}.py'''.format(model_name, datasets_name, model_name)
        subprocess.call(command, shell=True)
        print("Starter created!")

    @staticmethod
    def _clean_up(model_name):
        print("Clean up!")
        command = '''rm -rf training/train_start_{}.py'''.format(model_name)
        subprocess.call(command, shell=True)

    def train(self, model_name, datasets_name=None, gpu=0, clean_up=True):

        self._create_starter(model_name, datasets_name)

        if gpu == -1:
            _gpu_env_var = 'export CUDA_VISIBLE_DEVICES='
        else:
            _gpu_env_var = 'export CUDA_VISIBLE_DEVICES={}'.format(gpu)

        command = "& python -m training.train_start_{}".format(model_name)
        command = " ".join([_gpu_env_var, command])
        print("Commands: {}".format(command))
        subprocess.call(command, shell=True)
        if clean_up:
            self._clean_up(model_name=model_name)


if __name__ == '__main__':
    fire.Fire(Commander)
