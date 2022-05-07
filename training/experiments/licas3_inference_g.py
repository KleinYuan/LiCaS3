import os
import yaml
from box import Box
import tensorflow as tf
from ..utils.inference import create_inference_graph
from ..utils.freeze import freeze_graph_from_file
from ..utils.tf_ops import unet
import fire


class InferenceModel(object):
    """
    This is a modified graph based on training graph, usually:
            - Remove redundant operators
            - Replace placeholder of tf.bool with constant
            - Replace placeholder of is_training False
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_name = self.config.name
        self.placeholders_configs = self.config.tensors.placeholders
        self.hyper_params = self.config.tensors.hyper_params
        self.batch_size = self.config.train.batch_size
        self.num_frames = int(self.placeholders_configs.X.shape[-1] / 4)
        self.define_graph()

    def pre_process(self, x):
        """
        Converge X from shape (B, H, W, num_frames x 4) to (B x num_frames, H, W, 4) so that
        a network can learn to inference on single (?, H, W, 4)

        It also output an reduced tensor in the shape of (B , H, W, num_frames x 3 + 1) to avoid duplicate depth maps

        :param X:  self.X input (B, H, W, num_frames x 4)
        :return:
        """
        _, h, w, _ = x.get_shape().as_list()
        # (B, H, W, num_frames x 4) to (B x num_frames, H, W, 4)

        # Reshape (B , H, W, num_frames x 4) to (B , H, W, num_frames, 4)
        X_bhwf4 = tf.reshape(x, [-1, h, w, self.num_frames, 4], name='reshaped')
        # Transpose (B , H, W, num_frames, 4) to (B , num_frames, H, W, 4)
        X_bfhw4 = tf.transpose(X_bhwf4, [0, 3, 1, 2, 4], name='transposed')
        # Reshape (B, H, W, num_frames, 4) to (B x num_frames, H, W, 4)
        X_reorganized = tf.reshape(X_bfhw4, [-1, h, w, 4])

        with tf.name_scope("x_sandwitch"):
            X_single_frame = None
            for _c in range(0, self.num_frames):
                if X_single_frame is None:
                    X_single_frame = x[:, :, :, 0:3]
                else:
                    X_single_frame = tf.concat([X_single_frame, x[:, :, :, _c * 4: (_c * 4 + 3)]], -1)

            X_seqs = tf.concat([X_single_frame, x[:, :, :, 3:4]], -1, name="X_seqs")  # (B, H, W, 22)

        return X_reorganized, X_seqs

    def define_net(self):
        with tf.name_scope(self.model_name):
            with tf.variable_scope("Placeholders"):
                self.X = tf.placeholder(tf.float32, [None] + self.placeholders_configs.X.shape, name='X')  # (B, H, W, num_framesx 4)
                self.is_training = False

            with tf.variable_scope("pre_process"):
                # [cam, cam, cam, depth, cam, cam, cam, depth, ...]
                X_reorganized, X_seqs = self.pre_process(x=self.X)

            self.x_cam = tf.identity(X_reorganized[:, :, :, 0:3], name='x_cam')  # (B x num_frames, H, W, 3)
            self.x_dm = tf.identity(X_reorganized[:, :, :, 3:4], name='x_dm')  # (B x num_frames, H, W, 1)

            # Predicting depth map from camera
            self.hat_x_dm = unet(inputs=self.x_cam, num_classes=1, base_depth=self.hyper_params.base_depth, name='unet')


            with tf.variable_scope("Inference"):
                alpha = 0.85
                ssim_loss = tf.image.ssim(img1=self.x_dm, img2=tf.nn.sigmoid(self.hat_x_dm), max_val=1)
                l1_loss = tf.abs(self.x_dm - tf.nn.sigmoid(self.hat_x_dm))
                ph_construction_loss = alpha * 0.5 * (1 - ssim_loss) + (1 - alpha) * l1_loss
                ce_of_dms = tf.reduce_mean(ph_construction_loss, axis=[1, 2, 3])
            self.ce_of_dms_bf = tf.reshape(ce_of_dms, [-1, self.num_frames], name='losses_from_g')
            self.prediction = tf.argmin(self.ce_of_dms_bf, 1, name='prediction_from_g')

    def define_graph(self):
        self.logger.info('[InferenceModel] Constructing graph now...')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.define_net()


class InferenceCkptProcessor(object):

    def process(self, config_fp, from_dir, to_dir, to_name):
        config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
        config = Box(yaml.load(open(config_abs_fp, 'r').read()))
        # Config logger
        tf.logging.set_verbosity(tf.logging.INFO)
        logger = tf.logging

        inference_model = InferenceModel(config=config, logger=logger)
        logger.info("Create inference graph from {} to {}".format(from_dir, to_dir))
        create_inference_graph(inference_model.graph, from_dir, to_dir, to_name, config.inference.included_tensor_names)
        freeze_graph_from_file(to_dir, to_name, config.inference.freeze.output_node_name)


if __name__ == '__main__':
    fire.Fire(InferenceCkptProcessor)
