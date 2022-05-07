import tensorflow as tf
from ..core.base_model import BaseModel
from ..utils.tf_ops import resnet_v2_50


class Model(BaseModel):

    def init(self):
        self.model_name = self.config.name
        self.placeholders_configs = self.config.tensors.placeholders
        self.num_frames = int(self.placeholders_configs.X.shape[-1] / 4)
        self.hyper_params = self.config.tensors.hyper_params

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

            X_seqs = tf.concat([X_single_frame, x[:, :, :, 3:4]], -1, name="X_seqs") # (B, H, W, 22)

        return X_reorganized, X_seqs

    def define_net(self):
        with tf.name_scope(self.model_name):
            with tf.variable_scope("Placeholders"):
                self.X = tf.placeholder(tf.float32, [None] + self.placeholders_configs.X.shape, name='X') # (B, H, W, num_framesx 4)
                self.Y = tf.placeholder(tf.float32, [None] + self.placeholders_configs.Y.shape, name='Y') # (B, 1)
                self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.name_scope("flip_augmentation"):
                do_flip = tf.random.uniform([]) > 0.5
                self.X_aug = tf.cond(do_flip, lambda: tf.image.flip_left_right(self.X), lambda: self.X)
            self.X = tf.cond(self.is_training, lambda: self.X_aug, lambda: self.X)

            with tf.variable_scope("pre_process"):
                # [cam, cam, cam, depth, cam, cam, cam, depth, ...]
                X_reorganized, X_seqs = self.pre_process(x=self.X)

            # Predicting idx of the synchronized one
            feature = None
            latent_dim = None  # 16
            viz = None
            with tf.variable_scope("classifier"):
                _dm = X_seqs[:, :, :, self.num_frames * 3: self.num_frames * 3 + 1]
                for _f in range(0, self.num_frames):
                    _single_cam = X_seqs[:, :, :, _f * 3: (_f + 1) * 3]
                    _single_cam = tf.math.multiply(_single_cam, _dm)
                    _single_cam = tf.concat([_single_cam, _dm], -1)
                    _single_cam_feature, _ = resnet_v2_50(_single_cam, latent_dim, is_training=self.is_training,
                                                          reuse=tf.AUTO_REUSE, scope='cam_ft')  # (B, 1024)
                    if feature is None:
                        feature = _single_cam_feature
                        viz = _single_cam
                    else:
                        feature = tf.concat([feature, _single_cam_feature], -1)
                        viz = tf.concat([viz, _single_cam], 1)

                feature_flatten = tf.layers.flatten(feature)
                fc1 = tf.layers.dense(feature_flatten, 512, activation=tf.nn.leaky_relu)
                fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.leaky_relu)
                self.hat_y = tf.layers.dense(fc2, self.num_frames, activation=None)

    def define_tensor_dict(self):
        self.tensor_dict = {
            'X': self.X,
            'Y': self.Y,
            'is_training':
                {
                    'tensor': self.is_training,
                    'value': {
                        'train': True,
                        'inference': False
                    }
                }
        }

    def cal_classification_loss(self):
        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(tf.cast(self.Y, tf.int64), depth=int(self.num_frames)),
                logits=self.hat_y
            ))
        return classification_loss

    def define_loss(self):
        self.logger.info('Defining loss ...')
        with tf.variable_scope('loss'):
            self.classification_loss = self.cal_classification_loss() * float(self.hyper_params.classification_loss_weight)
            self.loss = self.classification_loss

    def get_reset_metrics_ops(self):
        return tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

    def get_train_ops(self):
        return {
            "train": [self.optimizer_stage1, self.optimizer_stage2, self.loss, self.train_acc_ce_classifier_ops, self.train_acc_ce_topk_classifier_ops],
            "test": [self.loss, self.test_acc_ce_classifier_ops, self.test_acc_ce_topk_classifier_ops]
        }

    @staticmethod
    def custom_metrics(labels, predictions, name):
        return tf.metrics.mean(tf.nn.in_top_k(predictions=predictions, targets=labels, k=2), name=name)

    def define_summary_list(self):
        with tf.name_scope('predictions'):
            # Direct prediction from the classifier
            predictions_from_resnet = tf.argmax(self.hat_y, 1, name='prediction_from_classifier')
        gt_sync_idx = tf.cast(tf.squeeze(self.Y, axis=-1), tf.int32)

        with tf.name_scope('metrics'):
            with tf.name_scope('train'):


                self.train_acc_ce_classifier, self.train_acc_ce_classifier_ops = tf.metrics.accuracy(
                    labels=gt_sync_idx,
                    predictions=predictions_from_resnet,
                    name='acc_ce_classifier'
                )
                self.train_acc_ce_classifier_topk, self.train_acc_ce_topk_classifier_ops = self.custom_metrics(
                    labels=gt_sync_idx,
                    predictions=self.hat_y,
                    name='acc_ce_classifier_topk'
                )

            with tf.name_scope('test'):


                self.test_acc_ce_classifier, self.test_acc_ce_classifier_ops = tf.metrics.accuracy(
                    labels=gt_sync_idx,
                    predictions=predictions_from_resnet,
                    name='acc_ce_classifier'
                )

                self.test_acc_ce_classifier_topk, self.test_acc_ce_topk_classifier_ops = self.custom_metrics(
                    labels=gt_sync_idx,
                    predictions=self.hat_y, # (B, F)
                    name='acc_ce_classifier_topk'
                )
        self.summary_list = [
            tf.summary.scalar("train/loss", self.loss),
            tf.summary.scalar("test/loss", self.loss, collections=["test"]),
            tf.summary.scalar("train/classification_loss", self.classification_loss),
            tf.summary.scalar("test/classification_loss", self.classification_loss, collections=["test"]),
            tf.summary.scalar('test/acc_ce_classifier', self.test_acc_ce_classifier, collections=["test"]),
            tf.summary.scalar('train/acc_ce_classifier', self.train_acc_ce_classifier),
            tf.summary.scalar('test/acc_ce_classifier_topk', self.test_acc_ce_classifier_topk, collections=["test"]),
            tf.summary.scalar("train/acc_ce_classifier_topk", self.train_acc_ce_classifier_topk)
        ]

    def define_optimizer(self):
        self.logger.info('[Cosine Restart] Defining Optimizer ...')
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_stage1 = self.start_lr
        self.learning_rate_stage2 = float(self.config.train.stage2_learning_rate)

        tf.summary.scalar("lr_stage1", self.learning_rate_stage1)
        tf.summary.scalar("lr_stage2", self.learning_rate_stage2)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer_stage1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate_stage1)
        optimizer_stage2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate_stage2)
        if self.config.train.continue_training:
            _var_list = [v for v in tf.trainable_variables() if
                         v.name.split('/')[0] in self.config.train.optimizer_var_list]
            self.logger.info('[Optimizing Var List] {}'.format(_var_list))
        else:
            _var_list = []
        if len(_var_list) == 0:
            self.logger.info('[Optimizing Var List] Optimizing all variables')
            _var_list = None
        optimizer_stage1 = optimizer_stage1.minimize(self.loss, var_list=_var_list, global_step=self.global_step)
        self.optimizer_stage1 = tf.group([optimizer_stage1, update_ops])

        optimizer_stage2 = optimizer_stage2.minimize(self.loss, var_list=_var_list, global_step=self.global_step)
        self.optimizer_stage2 = tf.group([optimizer_stage2, update_ops])
        self.optimizer = [self.optimizer_stage1, self.optimizer_stage2 ]
