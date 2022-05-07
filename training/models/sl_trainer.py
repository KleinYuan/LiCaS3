from ..core.base_trainer import BaseTrainer as BaseTrainerTemplate
import tensorflow as tf


class BaseTrainer(BaseTrainerTemplate):

    def pre_process(self, feed_dict, tensor_dict):
        feed_dict[tensor_dict['X']] = feed_dict[tensor_dict['X']] / 255.
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
        epoch = int(epoch)
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
                    if epoch <= int(self.config.train.first_stage_epochs):
                        stage = 1
                        _train_ops = [train_ops["train"][0], train_ops["train"][2]] + train_ops["train"][2:]
                    else:
                        stage = 2
                        _train_ops = [train_ops["train"][1], train_ops["train"][2]] + train_ops["train"][2:]
                    train_evals = self.session.run(_train_ops, feed_dict=feed_dict)
                    loss_train = train_evals[1]
                    summary = train_evals[-1]
                    _train_loss += loss_train
                    self.logger.info('  [Stage {} - {} th iteration] Train loss: {}'.format(stage, cnt_iter, loss_train))
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
