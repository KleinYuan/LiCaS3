from ..utils.tfrecords import get_iterator_from_tfrecords


class BaseDataGenerator(object):
    _train_batch_iterator = None
    _test_batch_iterator = None
    _train_batch_init_op = None
    _test_batch_init_op = None

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.load_data_tfrecords()

    def load_data_tfrecords(self):
        self._train_batch_iterator, self._train_batch_init_op = get_iterator_from_tfrecords(
            config=self.config,
            test=False)

        self._test_batch_iterator, self._test_batch_init_op = get_iterator_from_tfrecords(
            config=self.config,
            test=True)

    @property
    def train_batch_iterator(self):
        return self._train_batch_iterator

    @property
    def test_batch_iterator(self):
        return self._test_batch_iterator

    @property
    def train_batch_init_op(self):
        return self._train_batch_init_op

    @property
    def test_batch_init_op(self):
        return self._test_batch_init_op

