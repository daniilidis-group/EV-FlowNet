#!/usr/bin/env python
import tensorflow as tf

def build_resnet_block(input_res, channelsout, is_training, data_format,
                       do_batch_norm=False, name=None):
    inputs = tf.identity(input_res)
    out_res_1 = general_conv2d(input_res,
                               name=name+'_res1',
                               channelsout=channelsout,
                               strides=1,
                               do_batch_norm=do_batch_norm,
                               is_training=is_training,
                               data_format=data_format)
    out_res_2 = general_conv2d(out_res_1,
                               name=name+'_res2',
                               channelsout=channelsout,
                               strides=1,
                               do_batch_norm=do_batch_norm,
                               is_training=is_training,
                               data_format=data_format)
    return out_res_2 + inputs

def general_conv2d(conv, name=None, channelsout=64, ksize=3, strides=2, init_factor=0.1,
                   padding='SAME', do_batch_norm=False, activation=tf.nn.relu,
                   is_training=True, data_format=None):
    
    conv = tf.layers.conv2d(conv,
                            channelsout,
                            ksize,
                            strides=strides,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=\
                            tf.contrib.layers.variance_scaling_initializer(factor=init_factor),
                            bias_initializer=tf.constant_initializer(0.0),
                            data_format=data_format)

    if do_batch_norm:
        conv = tf.layers.batch_normalization(conv,
                                             axis=1 if data_format=='channels_first' else -1,
                                             epsilon=1e-5,
                                             gamma_initializer=tf.constant_initializer([0.01]),
                                             name=name+'_bn',
                                             training=is_training)
    return conv

"""
Upsample a tensor by a factor of 2 with fixed padding and then do normal conv2d on it. 
Similar operation to a transposed convolution, but avoids checkerboard artifacts.
"""
def upsample_conv2d(conv, name=None, channelsout=64, ksize=3, init_factor=0.1,
                    do_batch_norm=False, is_training=True, data_format=None):
    if data_format == 'channels_first':
        conv = tf.transpose(conv, [0,2,3,1])
        shape = tf.shape(conv)
        conv = tf.image.resize_images(conv, size=[shape[1]*2, shape[2]*2],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        conv = tf.pad(conv,
                      paddings = [[0,0],
                                  [(ksize-1)/2, (ksize-1)/2],
                                  [(ksize-1)/2, (ksize-1)/2],
                                  [0,0]],
                      mode = 'REFLECT')
        
        conv = tf.transpose(conv, [0,3,1,2])
    else:
        shape = tf.shape(conv)
        conv = tf.image.resize_images(conv, size=[shape[1]*2, shape[2]*2],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        conv = tf.pad(conv,
                      paddings = [[0,0],
                                  [(ksize-1)/2, (ksize-1)/2],
                                  [(ksize-1)/2, (ksize-1)/2],
                                  [0,0]],
                      mode = 'REFLECT')

    conv = general_conv2d(conv, name=name, channelsout=channelsout, ksize=ksize, strides=1,
                          do_batch_norm=do_batch_norm, padding='VALID', init_factor=init_factor,
                          is_training=is_training, data_format=data_format)
    
    return conv

def predict_flow(conv, name=None, channelsout=2, ksize=1, strides=1,
                 padding='SAME', init_factor=0.1,
                 is_training=True, data_format=None):
    conv = general_conv2d(conv,
                          channelsout=channelsout,
                          ksize=ksize,
                          strides=strides,
                          init_factor=init_factor,
                          padding=padding,
                          activation=tf.tanh,
                          is_training=is_training,
                          data_format=data_format,
                          name=name)
    return conv
