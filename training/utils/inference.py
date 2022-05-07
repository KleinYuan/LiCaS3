import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import pathlib


def get_restore_weights_fn(graph, model_dir=None, ckp_dir=None, included_tensor_names=None, excluded_tensor_names=None):
    """
    https://www.tensorflow.org/api_docs/python/tf/contrib/framework/get_variables_to_restore
    :param graph: the graph to be restored
    :param model_dir: folder include the checkpoint, .meta, ...
    :param dir: directory which you save the pretrained checkpoints: .meta, .data, .index
    :param included_tensor_names: the tensor names you wanna include to restore
    :param excluded_tensor_names: the tensor names you wanna exclude to restore
    :return: a function that's supposed to load all params into memory
    """
    print("[Restoring weights] Setting ......")
    print("[Restoring weights] Restoring from model_dir={}".format(model_dir))
    if model_dir is not None:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        print("[Restoring weights] checkpoint: {}".format(checkpoint))
        ckp_dir = checkpoint.model_checkpoint_path
        print("[Restoring from ] {}".format(ckp_dir))
    with graph.as_default():
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=included_tensor_names,
            exclude=excluded_tensor_names)
        print(variables_to_restore)
        print("[Restoring weights] Restore function setting ......")
        print("[Variables to restore]  {} variables".format(len(variables_to_restore)))
        print("[Model] Loading ....{}".format(ckp_dir))
        restore_graph_fn = tf.contrib.framework.assign_from_checkpoint_fn(ckp_dir, variables_to_restore,
                                                                          ignore_missing_vars=True)
    print("[Restoring weights] Restore function set ......")
    return restore_graph_fn


def create_inference_graph(graph, from_dir, to_dir, to_name, included_tensor_names):
    print("===== Inference Graph Stats ================")
    print("Creating inference graph from {} to {}".format(
        from_dir,
        to_dir))
    print("[Source Checkpoints     ]       {}".format(from_dir))
    print("[Destination Checkpoints]       {}".format(to_dir))
    print("[Graph Scope to Save    ]       {}".format(included_tensor_names))
    if not os.path.isdir(to_dir):
        print("{} does not exits, creating one.".format(to_dir))
        pathlib.Path(to_dir).mkdir(parents=True, exist_ok=True)

    print("===== Inference Graph Stats ================")
    print("Graph: {}".format(graph))
    with graph.as_default():
        # Start a session to load weights to new graph
        tf_config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            print("Initialization .......")
            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            print("There are {} variables intotal!".format(len(tf.global_variables())))
            print("Assigning weighs to graph .......")
            restore_graph_fn = get_restore_weights_fn(sess.graph,
                                                      model_dir=from_dir,
                                                      included_tensor_names=included_tensor_names,
                                                      excluded_tensor_names=None)

            restore_graph_fn(sess)
            save_path = '{}/{}'.format(to_dir, to_name)
            print("Saving to {} .......".format(save_path))
            saver.save(sess=sess, save_path=save_path)

        # TODO: Some report shall be done here such as graph size, inference speed, operator types, ...
        print("===== Job completed ========================")