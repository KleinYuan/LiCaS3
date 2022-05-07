import tensorflow as tf
from tensorflow.python.client import timeline


class PBServer(object):

    def __init__(self, config):
        tf.reset_default_graph()
        self.session = None
        self.in_progress = False
        self.saver = None
        self.graph = None
        self.prediction = None
        self.config = config
        self.feed_dict = {}
        self.output_ops = []
        self.input_ops = []

        # All configs contents shall be fetched here as instance properties
        self.input_tensor_names = config.tensors.inputs
        self.output_tensor_names = config.tensors.outputs
        self.pb_fp = self.config.pb_fp

        self._load_model()
        self._init_predictor()

    def _load_model(self):
        print("     Loading frozen protobuf from {} ...".format(self.pb_fp))
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.pb_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        tf.get_default_graph().finalize()

    def _init_predictor(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config=tf_config, graph=self.graph)
            self._fetch_tensors()

    def _fetch_tensors(self):
        assert len(self.input_tensor_names) > 0
        assert len(self.output_tensor_names) > 0
        for _tensor_name in self.input_tensor_names:
            print("     Fetching {}".format(_tensor_name))
            _op = self.graph.get_tensor_by_name(_tensor_name)
            self.input_ops.append(_op)
            self.feed_dict[_op] = None
            print("     Fetched {}".format(_tensor_name))
        for _tensor_name in self.output_tensor_names:
            print("     Fetching {}".format(_tensor_name))
            _op = self.graph.get_tensor_by_name(_tensor_name)
            self.output_ops.append(_op)
            print("     Fetched {}".format(_tensor_name))
        print("     Fetched Input Ops: {}".format(self.input_ops))
        print("     Fetched Output Ops: {}".format(self.output_ops))

    def _set_feed_dict(self, data):
        assert len(data) == len(self.input_ops), \
            "Shape of data [{}] does not match shape of input ops [{}]" \
                .format(len(data), len(self.input_ops))
        with self.graph.as_default():
            for ind, op in enumerate(self.input_ops):
                self.feed_dict[op] = data[ind]

    def inference(self, data, profile_export=None):
        self.in_progress = True
        with self.graph.as_default():
            self._set_feed_dict(data=data)
            if profile_export is None:
                self.prediction = self.session.run(self.output_ops, feed_dict=self.feed_dict)
            else:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                self.prediction = self.session.run(self.output_ops, feed_dict=self.feed_dict, options=run_options, run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('{}.json'.format(profile_export), 'w') as f:
                    f.write(ctf)
        self.in_progress = False

        return self.prediction
