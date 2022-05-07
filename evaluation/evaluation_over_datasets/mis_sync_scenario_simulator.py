from box import Box
import yaml
import fire
from copy import deepcopy
from newer_college_mis_sync_scenario_simulator import NewerCollegeMisSyncScenarioSimulator
from kitti_mis_sync_scenario_simulator import KITTIMisSyncScenarioSimulator


class Simulator(object):

    def run(self, benchmark_config_template_fp, models_config_fp, dataset_name, model_type, simulation_model_id=None,
            simulation_num_frames_in_seq=None):
        benchmark_config_template = Box(yaml.load(open(benchmark_config_template_fp, 'r').read())).benchmarks
        models_config = Box(yaml.load(open(models_config_fp, 'r').read()))

        model_count = 0
        if (simulation_model_id) and (simulation_model_id in models_config.models[dataset_name].keys()):
            models_config.models[dataset_name] = {
                simulation_model_id: models_config.models[dataset_name][simulation_model_id]}
            print("Only output the simulation for model : {}".format(simulation_model_id))
            print("models_config.models[dataset_name]: {}".format(models_config.models[dataset_name]))

        for model_id, test_model_info in models_config.models[dataset_name].items():
            benchmark_config_template_updated = deepcopy(benchmark_config_template)
            s = int(test_model_info.s)
            l = int(test_model_info.l)
            if 'reduce_lidar_line_to' in test_model_info:
                reduce_lidar_line_to = int(test_model_info.reduce_lidar_line_to)
            else:
                reduce_lidar_line_to = int(benchmark_config_template_updated.sensors_info.lidar.beams)
            model_fp = test_model_info.model_fp
            print("[{}  / {}] Testing model id {}, s = {}, l = {}\n   model path = {}".format(
                model_count, len(models_config.models[dataset_name]), model_id, s, l, model_fp))

            # TODO(kaiwen): I am assuming you always have a timestamp appended.
            #  Otherwise, you will need to change this line.
            model_namespace = model_fp.split('/')[-2].split('@')[0]

            benchmark_config_template_updated.pb_fp = model_fp
            benchmark_config_template_updated.model_type = model_type
            if model_type == 'g':
                benchmark_config_template_updated.tensors = deepcopy(benchmark_config_template_updated.g_tensors)
                for input_tensor_idx, _ in enumerate(benchmark_config_template_updated.tensors.inputs):
                    benchmark_config_template_updated.tensors.inputs[input_tensor_idx] = \
                        benchmark_config_template_updated.g_tensors.inputs[input_tensor_idx].format(model_namespace)
                for output_tensor_idx, _ in enumerate(benchmark_config_template_updated.tensors.outputs):
                    benchmark_config_template_updated.tensors.outputs[output_tensor_idx] = \
                        benchmark_config_template_updated.g_tensors.outputs[output_tensor_idx].format(model_namespace)
            else:
                benchmark_config_template_updated.tensors = deepcopy(benchmark_config_template_updated.h_tensors)
                for input_tensor_idx, _ in enumerate(benchmark_config_template_updated.tensors.inputs):
                    benchmark_config_template_updated.tensors.inputs[input_tensor_idx] = \
                        benchmark_config_template_updated.h_tensors.inputs[input_tensor_idx].format(model_namespace)
                for output_tensor_idx, _ in enumerate(benchmark_config_template_updated.tensors.outputs):
                    benchmark_config_template_updated.tensors.outputs[output_tensor_idx] = \
                        benchmark_config_template_updated.h_tensors.outputs[output_tensor_idx].format(model_namespace)

            benchmark_config_template_updated.training_data.sampling_window = l
            benchmark_config_template_updated.training_data.sampling_stride = s
            benchmark_config_template_updated.training_data.features.X.C = (l + 1) * 4
            benchmark_config_template_updated.training_data.reduce_lidar_line_to = reduce_lidar_line_to
            if simulation_num_frames_in_seq:
                models_config.models.variables.num_frames_in_seq = [simulation_num_frames_in_seq]
                print("simulation num frame in seq is set as : {}".format(simulation_num_frames_in_seq))
            for num_frames_in_seq in models_config.models.variables.num_frames_in_seq:
                benchmark_config_template_updated.training_data.multi_frame_test.num_frames_in_seq = num_frames_in_seq

                if dataset_name == 'newer_college_psudo_gt':
                    simulation_runner = NewerCollegeMisSyncScenarioSimulator()
                    _ = simulation_runner.batch_run_over_psudo_gt(config=benchmark_config_template_updated)
                elif dataset_name == 'kitti':
                    simulation_runner = KITTIMisSyncScenarioSimulator()
                    _ = simulation_runner.batch_run(config=benchmark_config_template_updated)
                else:
                    raise Exception("{} dataset is not supported!".format(dataset_name))

            model_count += 1


if __name__ == '__main__':
    fire.Fire(Simulator)
