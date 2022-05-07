import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(sess, graph, model_folder, meta_info, output_node_names, logger):
    """
    Freeze graph into single protobuf file.

    :param model_folder:            Where the model is stored
    :param meta_info:               Meta info to be appended to frozen graph name
    :param output_node_names:       '${node_name},${node_name},...'
    :param last_save_name:          Meta to be retrieved
    :param logger:                  Logger
    :return:
    """
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_graph_{}.pb".format(meta_info)
    input_graph_def = graph.as_graph_def()

    graph_def = sess.graph.as_graph_def()
    for node in graph_def.node:
        logger.debug("{}".format(node.name))
    # We use a built-in TF helper to export variables to constants
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        output_node_names.split(",")  # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    logger.info("{} ops in the final graph.".format(len(output_graph_def.node)))


def freeze_graph_from_file(model_folder, output_file_name, output_node_names):
    # We retrieve our checkpoint fullpath
    print("model_folder: {}".format(model_folder))
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/%s.pb" % output_file_name
    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        graph_def = sess.graph.as_graph_def()
        # for node in graph_def.node:
        #     print node.name
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

