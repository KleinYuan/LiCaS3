import os
import tensorflow as tf
import random
import pathlib
import time
import numpy as np


class BaseTrainer(object):
    """
    You shall implement
     - pre_process(), if nothing to pre_process, just return feed_dict
    """

    @staticmethod
    def _get_output_node_names(model_name, outputs):
        """
        By default, we set the output name as $model_name/$_output_name in config.
            - We force you to rename your outputs as graceful names so that they're readable
        :param model_name:
        :param outputs:
        :return:
        """
        output_node_names = ''
        for _output_name, _output in outputs.items():
            output_node_names += '{}/{},'.format(model_name, _output_name)
        return output_node_names[:-1]

    def _inspect_dependencies(self):
        """
        A monitor function to print out all dependencies of interests.
        """
        self.logger.info("Tensorflow Version: {}".format(tf.__version__))

    def __init__(self, compute_model, config, logger):
        self.ts = time.time()
        self.session = None
        self.writer = None
        self.writer_val = None
        self.last_save_name = None
        self.decrease_in_val_loss = np.inf
        self.min_val_loss = np.inf
        self.in_progress = False  # Mainly for unit/int-test
        self.logger = logger
        self._inspect_dependencies()
        self.compute_model = compute_model
        self.train_config = config.train
        self.stop_val_loss_decrease = self.train_config.stop_val_loss_decrease
        self.epochs = int(self.train_config.epochs)
        self.batch_size = int(self.train_config.batch_size)
        self.logdir = "{}/{}/{}/".format(config.machine.log_dir, config.name, self.ts)
        self.save_path = "{}/{}/{}/".format(config.machine.save_dir, config.name, self.ts)
        self.val_epoch = int(self.train_config.val_epoch)
        self.save_epoch = int(self.train_config.save_epoch)
        if not os.path.isdir(self.save_path):
            print("{} does not exits, creating one.".format(self.save_path))
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.logger.info(
            "[Stop Condition] Training will stop either epoch == {} or validation error decrease < {}".format(
                self.epochs, self.stop_val_loss_decrease))

    def _get_tf_config(self):
        """
        Define Session configuration
        :return: tf_config
        """

        tf_config = tf.ConfigProto(allow_soft_placement=True, device_count=self.config.train.devices)
        tf_config.gpu_options.allow_growth = True
        return tf_config

    def pre_process(self, feed_dict, tensor_dict):
        """
        If you don't need to do any pre-process, return feed_dict
        :param feed_dict:
        :param tensor_dict:
        :return: feed_dict
        """
        raise NotImplementedError

    def get_feed_dict(self, next_batch, tensor_dict, mode='train'):
        """
        :param next_batch:  tensorflow operators to retrieve value from tfrecords
        :param tensor_dict: a dictionary from compute model including mappings between tensor name and tensors
        :param mode:        'train' or 'inference'
        :return:            dictionary of feed_dict
        """
        assert 'inputs' in self.config.data, "Misconfiguration, inputs not in data"
        _feature_ls = self.config.data.inputs.keys()
        _batch_ops_ls = []
        feed_dict = {}
        # First, loop through all feature from tfrecords
        for _feature_scope in _feature_ls:
            _feature_names = self.config.data.inputs[_feature_scope].feature_names
            _feature_name = random.choice(_feature_names)
            # Then, construct data retrieve ops
            _batch_ops_ls.append(next_batch[_feature_name])
        _batch_ls = self.session.run(_batch_ops_ls)

        # Then, construct feed_dict include all features from tfrecords
        for _idx, _feature in enumerate(_feature_ls):
            assert _feature in tensor_dict, "tensor_dict does not include the key"
            feed_dict[tensor_dict[_feature]] = _batch_ls[_idx]

        # Last, add hyper-param placeholder values such as dropout, is training ...
        for _tensor_name, _content in tensor_dict.items():
            if type(_content) == dict:
                feed_dict[_content['tensor']] = _content['value'][mode]
        return feed_dict

    def run_one_epoch(self, epoch, batch_init, next_batch, train_ops, is_training=True):
        """
        This function is define to run one tf.session.run, either training/inference
        We assume:
            - The train_ops is static and if you need to dynamically switch it, you shall consider
              override this function
            - You will have to implement pre_process function
            -
        :param epoch:                Int
        :param next_batch:           Tensor
        :param train_ops:            Array of Ops by default [optimizer, loss, summary_op] in prepare_train function
        :param is_training:          Python Boolean
        :return:
        """
        cnt_iter = 0
        _train_loss = 0
        _val_loss = 0
        summary = None
        summary_val = None
        if ('one_shot' not in self.config.data) or (not self.config.data.one_shot):
            self.session.run(batch_init)
        try:
            while True:
                cnt_iter += 1
                mode = 'train' if is_training else 'inference'
                feed_dict = self.get_feed_dict(
                    next_batch=next_batch,
                    tensor_dict=self.compute_model.tensor_dict,
                    mode=mode)
                feed_dict = self.pre_process(feed_dict=feed_dict, tensor_dict=self.compute_model.tensor_dict)
                if is_training:
                    train_evals = self.session.run(train_ops["train"], feed_dict=feed_dict)
                    loss_train = train_evals[1]
                    summary = train_evals[-1]
                    _train_loss += loss_train
                    self.logger.info('  [{} th iteration] Train loss: {}'.format(cnt_iter, loss_train))
                else:
                    test_evals = self.session.run(train_ops["test"], feed_dict=feed_dict)
                    loss_test = test_evals[0]
                    summary_val = test_evals[-1]
                    self.logger.info('  [{} th iteration] Test loss: {}'.format(cnt_iter, loss_test))
                    _val_loss += loss_test

        except tf.errors.OutOfRangeError as e:
            self.logger.warn("End of data .")
            if is_training:

                _avg_loss = _train_loss / cnt_iter
                if summary is not None:
                    self.writer.add_summary(summary, epoch)
                self.logger.info('[{} th epoch] Average train loss: {}'.format(epoch, _avg_loss))
            else:
                _avg_loss = _val_loss / cnt_iter
                self.decrease_in_val_loss = self.min_val_loss - _avg_loss
                if _avg_loss < self.min_val_loss:
                    self.min_val_loss = _avg_loss
                    snapshot_path = self.saver.save(sess=self.session, save_path="{}best".format(self.save_path))
                    self.logger.info('[{} th epoch]: saving the best model to {}'.format(epoch, snapshot_path))
                if summary_val is not None:
                    self.writer_val.add_summary(summary_val, epoch)
                self.logger.info('[{} th epoch]: Average validation loss: {}'.format(epoch, _avg_loss))
            pass

    def prepare_train(self, data_generator, restore_graph_fns):
        """
        This function is a very generic function that
            - init graph
            - retrieve tfrecords init/batch get next ops
            - define saver
            - define summary ops
            - restore pre-trained model
        Mostly, you don't need to override.
        However, if you need to dynamically switch train_ops. you should do that after.
        :param data_generator:
        :param restore_graph_fns:  []
        :return:
        """
        # Initialize graph
        self.session.run(self.compute_model.init_graph)

        # Restore parameters from pre-trained graph
        self.logger.info("restore_graph_fns: {}".format(restore_graph_fns))
        # Saving GPU memory
        with tf.device("/cpu:0"):
            if len(restore_graph_fns) > 0:
                self.logger.info('Assign params to target tensor ...')
                for _restore_graph_fn in restore_graph_fns:
                    _restore_graph_fn(self.session)

        # Fetch iterators/init_ops of data
        train_batch_iterator = data_generator.train_batch_iterator
        test_batch_iterator = data_generator.test_batch_iterator
        train_batch_init_op = data_generator.train_batch_init_op
        test_batch_init_op = data_generator.test_batch_init_op
        next_train_batch = train_batch_iterator.get_next()
        next_test_batch = test_batch_iterator.get_next()

        # Initialize Saver, Summary Ops and train Ops
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        summary_op = tf.summary.merge_all()
        summary_op_test = tf.summary.merge_all("test")
        train_ops = {
            "train": self.compute_model.get_train_ops()["train"] + [summary_op],
            "test": self.compute_model.get_train_ops()["test"] + [summary_op_test]
        }

        self.writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.session.graph)
        self.writer_val = tf.summary.FileWriter(logdir=self.logdir + 'val/', graph=self.session.graph)
        return train_batch_init_op, test_batch_init_op, next_train_batch, next_test_batch, train_ops, self.saver

    def train(self, data_generator, restore_graph_fns):

        self.logger.info('Opening a session and training started ...')

        with tf.Session(graph=self.compute_model.graph, config=self._get_tf_config()) as self.session:
            self.in_progress = True
            train_batch_init_op, test_batch_init_op, next_train_batch, next_test_batch, train_ops, self.saver = \
                self.prepare_train(
                    data_generator=data_generator,
                    restore_graph_fns=restore_graph_fns
                )
            self.logger.info("Graph size at beginning of epoch : {}".format(self.session.graph_def.ByteSize()))

            should_early_stop = ('early_stop' in self.config.train) and self.config.train.early_stop
            # Train
            for _epoch in range(self.epochs):
                if should_early_stop and (self.decrease_in_val_loss <= self.stop_val_loss_decrease):
                    self.logger.info("Validation error reached configured one, stop!")
                    break
                self.logger.info('{} / {} th epoch, training ...'.format(_epoch, self.epochs))

                # Run training for every epoch
                self.run_one_epoch(
                    epoch=_epoch,
                    batch_init=train_batch_init_op,
                    next_batch=next_train_batch,
                    train_ops=train_ops,
                    is_training=True
                )
                if ('one_shot' in self.config.data) and self.config.data.one_shot:
                    next_train_batch = train_batch_init_op.make_one_shot_iterator().get_next()

                # Run testing for every several epoch
                if (_epoch > 0) and (_epoch % self.val_epoch == 0):
                    self.run_one_epoch(
                        epoch=_epoch,
                        batch_init=test_batch_init_op,
                        next_batch=next_test_batch,
                        train_ops=train_ops,
                        is_training=False
                    )
                    if ('one_shot' in self.config.data) and self.config.data.one_shot:
                        next_test_batch = test_batch_init_op.make_one_shot_iterator().get_next()
                # Save
                if (_epoch % self.save_epoch == 0) or (_epoch == self.epochs - 1):
                    self.last_save_name = "{}eps_{}_".format(self.save_path, _epoch)
                    snapshot_path = self.saver.save(sess=self.session, save_path=self.last_save_name)
                    self.logger.info('Snapshot of {} th epoch is saved to {}'.format(_epoch, snapshot_path))
                    self.logger.debug("Graph size at end of epoch : {}".format(self.session.graph_def.ByteSize()))
                reset_metrics_ops = self.compute_model.get_reset_metrics_ops()
                if reset_metrics_ops:
                    self.session.run(reset_metrics_ops)
                    self.logger.info("Reset the metrics Ops")

            self.logger.info('Training ended and model file is in here: {}'.format(self.save_path))
