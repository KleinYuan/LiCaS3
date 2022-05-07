import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2


def pool(input, size):
    # Only data_format='NHWC' is supported
    print('pool input =  {}'.format(input.get_shape()))
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def upsampling(input, factor):
    return tf.image.resize_nearest_neighbor(input, (tf.shape(input)[1] * factor, tf.shape(input)[2] * factor))


def conv_relu(input, kernel_size, depth, stride=1, activate=True, subscope='0', padding='SAME', initializer=tf.contrib.layers.xavier_initializer()):
    # Only data_format='NHWC' is supported
    print('conv_relu input =   {}'.format(input.get_shape()))
    weights = tf.get_variable('weights/{}'.format(subscope), shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer=initializer)
    biases = tf.get_variable('biases/{}'.format(subscope), shape=[depth], initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding=padding)
    if activate:
        return tf.nn.relu(conv + biases)
    else:
        return conv + biases


# This is not used but put here for reference, in case you would like to explore smaller model
def resnet_v2_18(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 include_root_block=True,
                 spatial_squeeze=True,
                 scope='resnet_v2_18',
                 reduction=1):
  resnet_v2_block = resnet_v2.resnet_v2_block
  blocks = [
      resnet_v2_block('block1', base_depth=64//reduction, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=128//reduction, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=256//reduction, num_units=2, stride=2),
      resnet_v2_block('block4', base_depth=512//reduction, num_units=2, stride=1),
  ]
  return resnet_v2.resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=include_root_block,
      # spatial_squeeze=spatial_squeeze,
      reuse=reuse,
      scope=scope)


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 include_root_block=True,
                 spatial_squeeze=True,
                 scope='resnet_v2_50',
                 reduction=1):
    return resnet_v2.resnet_v2_50(inputs,
      num_classes,
      is_training,
      reuse=reuse,
      scope=scope)


def unet(inputs, num_classes, base_depth, name):
    print("Start UNet ----- {} --------".format(name))
    with tf.compat.v1.variable_scope(name):
        with tf.compat.v1.variable_scope('conv1'):
            conv1 = conv_relu(input=inputs, kernel_size=3, depth=base_depth, subscope='1')
            conv1 = conv_relu(input=conv1, kernel_size=3, depth=base_depth, subscope='2')
            pool1 = pool(input=conv1, size=2)

        with tf.compat.v1.variable_scope('conv2'):
            conv2 = conv_relu(input=pool1, kernel_size=3, depth=2 * base_depth, subscope='1')
            conv2 = conv_relu(input=conv2, kernel_size=3, depth=2 * base_depth, subscope='2')
            pool2 = pool(conv2, size=2)

        with tf.compat.v1.variable_scope('conv3'):
            conv3 = conv_relu(input=pool2, kernel_size=3, depth=4 * base_depth, subscope='1')
            conv3 = conv_relu(input=conv3, kernel_size=3, depth=4 * base_depth, subscope='2')
            pool3 = pool(conv3, size=2)

        with tf.compat.v1.variable_scope('conv4'):
            conv4 = conv_relu(input=pool3, kernel_size=3, depth=8 * base_depth, subscope='1')
            conv4 = conv_relu(input=conv4, kernel_size=3, depth=8 * base_depth, subscope='2')
            pool4 = pool(conv4, size=2)

        with tf.compat.v1.variable_scope('conv5'):
            conv5 = conv_relu(input=pool4, kernel_size=3, depth=16 * base_depth, subscope='1')
            conv5 = conv_relu(input=conv5, kernel_size=3, depth=16 * base_depth, subscope='2')
            up_conv5 = upsampling(input=conv5, factor=2)

        with tf.compat.v1.variable_scope('conv6'):
            concated6 = tf.concat([conv4, up_conv5], -1)
            conv6 = conv_relu(input=concated6, kernel_size=3, depth=8 * base_depth, subscope='1')
            conv6 = conv_relu(input=conv6, kernel_size=3, depth=4 * base_depth, subscope='2')
            up_conv6 = upsampling(input=conv6, factor=2)

        with tf.compat.v1.variable_scope('conv7'):
            concated7 = tf.concat([conv3, up_conv6], -1)
            conv7 = conv_relu(input=concated7, kernel_size=3, depth=4 * base_depth, subscope='1')
            conv7 = conv_relu(input=conv7, kernel_size=3, depth=2 * base_depth, subscope='2')
            up_conv7 = upsampling(input=conv7, factor=2)

        with tf.compat.v1.variable_scope('conv8'):
            concated8 = tf.concat([conv2, up_conv7], -1)
            conv8 = conv_relu(input=concated8, kernel_size=3, depth=2 * base_depth, subscope='1')
            conv8 = conv_relu(input=conv8, kernel_size=3, depth=base_depth, subscope='2')
            up_conv8 = upsampling(input=conv8, factor=2)

        with tf.compat.v1.variable_scope('conv9'):
            concated9 = tf.concat([conv1, up_conv8], -1)
            conv9 = conv_relu(input=concated9, kernel_size=3, depth=base_depth, subscope='1')
            conv9 = conv_relu(input=conv9, kernel_size=3, depth=base_depth, subscope='2')

        with tf.compat.v1.variable_scope('conv10'):
            outputs = conv_relu(input=conv9, kernel_size=1, depth=num_classes, activate=False)
    print("End UNet ----- {} --------".format(name))
    return outputs
