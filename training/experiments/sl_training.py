import os
import yaml
from box import Box
import tensorflow as tf
from ..models.sl_model import Model
from ..models.sl_trainer import BaseTrainer
from ..core.base_data_generator import BaseDataGenerator


def run(config_fp='../configs/sl.yaml'):
    """
    This is a base training app.
    In most cases, you don't need to modify any lines of this script.
    """
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    # Config logger
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = tf.logging
    # Initialize Four Modules: Data, Trainer, Net, Graph
    compute_model = Model(config=config, logger=logger)
    trainer = BaseTrainer(compute_model=compute_model, config=config, logger=logger)

    # Run Training
    with compute_model.graph.as_default():
        # loading tfrecords need tf operations in the same graph
        data_generator = BaseDataGenerator(config=config)

    trainer.train(data_generator=data_generator, restore_graph_fns=compute_model.restore_graph_fns)