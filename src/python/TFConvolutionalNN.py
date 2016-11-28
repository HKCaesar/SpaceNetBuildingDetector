import tensorflow as tf

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

def build_inference(layers, filters, filter_size, scale, batch_size):
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
    x8, x3, label_distance = placeholder_inputs(FLAGS.ws, scale)

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

        labelout[i] = tf.nn.conv2d_transpose( z[i], Wo[i], [batch_size, ws, ws, 1] ,strides=[1,1,1,1], padding='SAME')
        if 0 == i:
            outlayer[i] = labelout[i]
        else :
            # convex combination measures impact of layer
            outlayer[i] = tf.nn.relu( tf.add(  tf.scalar_mul( beta[i] , labelout[i]), tf.scalar_mul(1.0-beta[i], inlayer[i])))

        label_cost[i] = tf.reduce_sum ( tf.pow( tf.sub(outlayer[i],label_distance),2))
        label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i], var_list=vars_Wb)
        full_label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i])

    return label_cost, label_optimizer, full_label_optimizer
