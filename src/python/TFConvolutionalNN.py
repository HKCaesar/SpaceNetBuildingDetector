import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('layers', 9, 'Number of layers.')
flags.DEFINE_integer('filters', 32, 'Number of filters/features per convolution')
flags.DEFINE_integer('filter_size', 5, 'Number of convolution filter\'s patch.')
flags.DEFINE_integer('ws', 100, 'Size of 8-band image.')
flags.DEFINE_integer('scale', 4, 'Scale to be applied to 8-band input image size in order to meet 3-band input image size')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('max_layer_steps', 100, 'Maximal number of steps per layer during training.')
flags.DEFINE_string('data_dir', 'data', 'Directory to get/put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
#flags.DEFINE_float('layers', 9, 'Number of layers.')

def placeholder_inputs(ws, scale):
    """"
    Creates placeholders for processing
    Args:
        ws: the edge size of 8-band input image (assuming square)
        scale: the scale to be applied to 8-band input image size in order to meet 3-band input image size
    Returns:
        x8: the placeholder for 8-band inputs
        x3: the placeholder for 3-band inputs
        label_distance: the placeholder for distance transform as a label
    """"
    x8 = tf.placeholder(tf.float32, shape=[None, ws, ws, 8]) # 8-band input
    x3 = tf.placeholder(tf.float32, shape=[None, scale * ws, scale * ws, 3]) # 3-band ipnput
    label_distance = tf.placeholder(tf.float32, shape=[None, ws, ws, 1]) # distance transform as a label

    return x8, x3, label_distance

def inference(x8, x3, label_distance, layers, filters, filter_size, scale, batch_size):
    """
    Build convolutional NN inference
    Args:
        layers: the number of layers for CNN to build
        filters: the number of filters/features per convolution
        filter_size: the size of convolution filter's patch
        scale: the scale to be applied to 8-band input image size in order to meet 3-band input image size
        batch_size: the batch size for processing
    Returns:
        label_cost: the cost per label
        label_optimizer: the reduced label optimizer
        full_label_optimizer: the full label optimizer operation pointer
    """
    #Generator
    for i in range(layers):
        alpha[i] = tf.Variable(0.9, name='alpha_' + str(i))
        beta[i] = tf.maximum( 0.0 , tf.minimum ( 1.0 , alpha[i] ), name='beta_'+str(i))
        bi[i] = tf.Variable(tf.constant(0.0,shape=[filters]), name='bi_'+str(i))
        bo[i] = tf.Variable(tf.constant(0.0,shape=[filters]), name='bo_'+str(i))
        Wo[i] = tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, filters], stddev=0.1), name='Wo_'+str(i))  #
        if 0 == i:
            # First layer project 11 bands onto one distance transform band
            Wi3 = tf.Variable(tf.truncated_normal([filter_size, filter_size, 3, filters], stddev=0.1), name='Wi_'+str(i)+'l3')
            Wi8 = tf.Variable(tf.truncated_normal([filter_size, filter_size, 8, filters], stddev=0.1), name='Wi_'+str(i)+'l8')
            z3 = tf.nn.conv2d( x3, Wi3, strides=[1, scale, scale,1], padding='SAME')
            z8 = tf.nn.conv2d( x8, Wi8, strides=[1,1,1,1], padding='SAME')
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.add(z3, z8), bi[i], name='conv_'+str(i))), bo[i])
            vars_Wb = [Wi3,Wi8,Wo[i],bi[i],bo[i]]
        else:
            # non-initial bands are perturbations of previous bands output
            inlayer[i] = outlayer[i-1]
            Wi[i] = tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, filters], stddev=0.1), name='Wi_'+str(i))
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d( inlayer[i], Wi[i], strides=[1,1,1,1], padding='SAME'), bi[i], name='conv_'+str(i))), bo[i])
            vars_Wb = [Wi[i],Wo[i],bi[i],bo[i], alpha[i]]

        labelout[i] = tf.nn.conv2d_transpose(z[i], Wo[i], [batch_size, ws, ws, 1], strides=[1,1,1,1], padding='SAME')
        if 0 == i:
            outlayer[i] = labelout[i]
        else :
            # convex combination measures impact of layer
            outlayer[i] = tf.nn.relu(tf.add(tf.scalar_mul(beta[i], labelout[i]), tf.scalar_mul(1.0-beta[i], inlayer[i])))

        label_cost[i] = tf.reduce_sum(tf.pow( tf.sub(outlayer[i], label_distance),2))# loss
        label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i], var_list=vars_Wb)
        full_label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i])

    return label_optimizer, full_label_optimizer

def loss(out_label_distance, label_distance):
    label_cost = tf.reduce_sum(tf.pow( tf.sub(out_label_distance, label_distance),2))
    return label_cost

def fill_feed_dict(x8_pl, x3_pl, label_distance_pl):
    random_input8, random_input3, random_label = GetRandomInput(labels, im_raw8, im_raw3, FLAGS.batch_size, FLAGS.ws, FLAGS.scale)
    feed_dict = {
        x8_pl: random_input8,
        x3_pl: random_input3,
        label_distance_pl: random_label
    }
    return feed_dict

def run_training():

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Generate placeholders for the images and distance labels.
        x8_placeholder, x3_placeholder, label_distance_placeholder = placeholder_inputs(FLAGS.ws, FLAGS.scale)

        # Add to the Graph the Ops that calculate and apply gradients.
        label_optimizer, full_label_optimizer = train(FLAGS.layers, FLAGS.filters, FLAGS.filter_size, FLAGS.scale, FLAGS.batch_size)

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # And then after everything is built:

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # Run the Op to initialize the variables.
            sess.run(init)

            global_step = 0
            for i in range(layers):
                for step in range(FLAGS.max_layer_steps):
                    global_step += 1
                    # Fill a feed dictionary with the actual set of images and labels for this particular training step.
                    feed_dict = fill_feed_dict()
                    sess.run(label_optimizer[i], feed_dict=feed_dict)
                for step in range(5 * FLAGS.max_layer_steps):
                    global_step += 1
                    # Fill a feed dictionary with the actual set of images and labels for this particular training step.
                    feed_dict = fill_feed_dict()
                    sess.run(full_label_optimizer[i], feed_dict=feed_dict)
                saver.save(sess, './checkpoint/boundary_'+str(i), global_step=global_step)
            saver.save(sess, './checkpoint/boundary', global_step=global_step)

def load_data_set():
    """
    Loads data set meta info
    
    """

def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
