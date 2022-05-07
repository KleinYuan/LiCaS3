from box import Box
import yaml
import fire
from copy import deepcopy
from kitti_runner import KITTIBenchmarks
from newer_college_gt_runner import NewerCollegeBenchmarksOverGT
from newer_college_psudo_gt_runner import NewerCollegeBenchmarksOverPsudoGT
from utils.metrics import get_metrics_formatted
import uuid
import os.path


class BatchBenchmarks(object):

    def run(self, benchmark_config_template_fp, models_config_fp, dataset_name, model_type):
        benchmark_config_template = Box(yaml.load(open(benchmark_config_template_fp, 'r').read())).benchmarks
        models_config = Box(yaml.load(open(models_config_fp, 'r').read()))

        model_count = 0
        for model_id, test_model_info in models_config.models[dataset_name].items():
            benchmark_config_template_updated = deepcopy(benchmark_config_template)
            s = int(test_model_info.s)
            l = int(test_model_info.l)
            if 'reduce_lidar_line_to' in test_model_info:
                reduce_lidar_line_to = int(test_model_info.reduce_lidar_line_to)
            else:
                reduce_lidar_line_to = int(benchmark_config_template_updated.sensors_info.lidar.beams)
            model_fp = test_model_info.model_fp
            print("[{}  / {}] Testing model id {}, s = {}, l = {}, lines = {} \n   model path = {}".format(
                model_count, len(models_config.models[dataset_name]), model_id, s, l, reduce_lidar_line_to, model_fp))

            # TODO(kaiwen): I am assuming you always have a timestamp appended.
            #  Otherwise, you will need to change this line.
            model_namespace = model_fp.split('/')[-2].split('@')[0]

            benchmark_config_template_updated.pb_fp = model_fp
            benchmark_config_template_updated.model_type = model_type
            # TODO: the tensor section is generated on the fly instead
            if model_type == 'g':
                benchmark_config_template_updated.tensors = deepcopy(benchmark_config_template_updated.g_tensors)
                benchmark_config_template_updated.tensors.inputs[0] = benchmark_config_template_updated.g_tensors.inputs[0].format(model_namespace)
                benchmark_config_template_updated.tensors.outputs[0] = benchmark_config_template_updated.g_tensors.outputs[0].format(model_namespace)
            else:
                benchmark_config_template_updated.tensors = deepcopy(benchmark_config_template_updated.h_tensors)
                benchmark_config_template_updated.tensors.inputs[0] = \
                benchmark_config_template_updated.h_tensors.inputs[0].format(model_namespace)
                benchmark_config_template_updated.tensors.outputs[0] = \
                benchmark_config_template_updated.h_tensors.outputs[0].format(model_namespace)

            benchmark_config_template_updated.training_data.sampling_window = l
            benchmark_config_template_updated.training_data.sampling_stride = s
            benchmark_config_template_updated.training_data.features.X.C = (l + 1) * 4
            benchmark_config_template_updated.training_data.reduce_lidar_line_to = reduce_lidar_line_to
            for num_frames_in_seq in models_config.models.variables.num_frames_in_seq:
                benchmark_config_template_updated.training_data.multi_frame_test.num_frames_in_seq = num_frames_in_seq

                if dataset_name == 'kitti':
                    benchmark_runner = KITTIBenchmarks()
                    metrics = benchmark_runner.batch_run(config=benchmark_config_template_updated)
                elif dataset_name == 'newer_college_gt':
                    benchmark_runner = NewerCollegeBenchmarksOverGT()
                    metrics = benchmark_runner.batch_run_over_gt(config=benchmark_config_template_updated)
                elif dataset_name == 'newer_college_psudo_gt':
                    benchmark_runner = NewerCollegeBenchmarksOverPsudoGT()
                    metrics = benchmark_runner.batch_run_over_psudo_gt(config=benchmark_config_template_updated)
                else:
                    raise Exception("{} dataset is not supported!".format(dataset_name))
                report_metrics = get_metrics_formatted(metrics)
                report_file_names = "{}/id{}_{}_l{}_s{}_seq_{}_{}.txt".format(models_config.output_folder, model_id, dataset_name, l, s, num_frames_in_seq,  uuid.uuid4().hex)
                f = open(report_file_names, "w")
                f.write("{}".format(metrics))
                f.close()

                consolidated_report_file_names = "{}/{}_consolidated.txt".format(models_config.output_folder, dataset_name)
                if os.path.isfile(consolidated_report_file_names):
                    f2 = open(consolidated_report_file_names, "a")
                else:
                    f2 = open(consolidated_report_file_names, "w")
                f2.write("\n{} | l = {}, s = {}, seq= {} |  {}".format(model_id, l ,s , num_frames_in_seq, report_metrics))
                f2.close()

            model_count += 1


if __name__ == '__main__':
    fire.Fire(BatchBenchmarks)
