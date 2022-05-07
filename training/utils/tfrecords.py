import os
import tensorflow as tf


def record_parser(record, feature_set):
    """
    Parse single example
    :param record:        Single record in tfrecord
    :param feature_set:   Dictionary of Features
    :return:              Parsed feature
    """
    features = tf.parse_single_example(record, features=feature_set)
    return features


def walk_through_data_path(data_dir, suffix):
    print("Scanning data ....")
    response = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(suffix):
                response.append(root + '/' + file)
    print("Scanned {} data in total!".format(len(response)))
    return response


def get_iterator_from_tfrecords(config, test=False):
    """
    !!! Important !!!
    All features shall be serialized in tf.float32
    with FixedLenFeature
    :param config:  Config box instance
    :param test:    Flag on whether it's a test
    :return:        iterator and init_op
    """

    # The nested for loop and if else is on purpose so that you can understand it
    # without dive deep into the not well documented tfrecords format

    feature_set = {}
    if 'num_parallel_reads' not in config.data:
        num_parallel_reads = 4
    else:
        num_parallel_reads = config.data.num_parallel_reads

    if ('one_shot' in config.data) and config.data.one_shot:
        one_shot = True
    else:
        one_shot = False

    if test:
        data_dirs = config.data.tfrecords_test_dirs
    else:
        data_dirs = config.data.tfrecords_train_dirs

    filenames = []
    for _data_dir in data_dirs:
        _filenames = walk_through_data_path(_data_dir, suffix=config.data.suffix)
        filenames = filenames + _filenames

    if test:
        print("Loading testing tfrecords: {}".format(filenames))
    else:
        print("Loading training tfrecords: {}".format(filenames))

    for _, _input in config.data.inputs.items():
        if _input.modality == 'image':
            _shape = (_input.H, _input.W, _input.C) if _input.nhwc else (_input.C, _input.H, _input.W)
        else:
            _shape = _input.shape
        for _feature_name in _input.feature_names:
            if ('var_len' in _input) and _input.var_len:
                feature_set.update({
                    _feature_name: tf.VarLenFeature(dtype=getattr(tf.dtypes, _input.data_type))
                })
            else:
                feature_set.update({
                    _feature_name: tf.FixedLenFeature(shape=_shape, dtype=getattr(tf.dtypes, _input.data_type))
                })

    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type=config.data.compression_type,
        num_parallel_reads=num_parallel_reads
    )

    dataset = dataset.map(lambda x: record_parser(x, feature_set))

    if one_shot:
        iterator = dataset.make_one_shot_iterator()
        return iterator, dataset
    else:
        dataset = dataset.batch(config.train.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        init_op = iterator.make_initializer(dataset)
        return iterator, init_op
