from box import Box
import fire
import yaml


class Generator(object):

    def generate_training_data(self, config, *args, **kwargs):
        """
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
        """
        training_examples_data_info = []
        raise NotImplementedError

    def serialize_data_into_tfrecords(self, config, training_examples_data_info, *args, **kwargs):
        raise NotImplementedError

    def run(self, config_fp, *args, **kwargs):
        print("Step1: Loading configuration file ...")
        config = Box(yaml.load(open(config_fp, 'r').read()))

        print("Step2: Generate training data ...")
        training_examples_data_info = self.generate_training_data(config=config)

        print("Step3: Serialize data into tfrecords ...")
        self.serialize_data_into_tfrecords(config=config, training_examples_data_info=training_examples_data_info)


if __name__ == '__main__':
    fire.Fire(Generator)
