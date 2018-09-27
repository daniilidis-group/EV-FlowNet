#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from basic_layers import *

_BASE_CHANNELS = 64

def encoder(inputs, is_training, data_format, do_batch_norm=False):
    skip_connections = {}
    with tf.variable_scope('encoder'):
        for i in range(4):
            inputs = general_conv2d(inputs,
                                    name='conv{}'.format(i),
                                    channelsout=(2**i)*_BASE_CHANNELS,
                                    do_batch_norm=do_batch_norm,
                                    is_training=is_training,
                                    data_format=data_format)
            skip_connections['skip{}'.format(i)] = inputs

    return inputs, skip_connections

def transition(inputs, is_training, data_format, do_batch_norm=False):
    with tf.variable_scope('transition'):
        for i in range(2):
            inputs = build_resnet_block(inputs,
                                        channelsout=8*_BASE_CHANNELS,
                                        is_training=is_training,
                                        do_batch_norm=do_batch_norm,
                                        data_format=data_format,
                                        name='res{}'.format(i))
    return inputs

def decoder(inputs, skip_connection, is_training, data_format, do_batch_norm=False):   
    with tf.variable_scope('decoder'):
        flow_dict = {}
        for i in range(4):
            # Skip connection.
            inputs = tf.concat([inputs, skip_connection['skip{}'.format(3-i)]],
                               axis=1 if data_format=='channels_first' else -1)
            
            inputs = upsample_conv2d(inputs,
                                     name='deconv{}'.format(i),
                                     channelsout=(2**(2-i))*_BASE_CHANNELS,
                                     do_batch_norm=do_batch_norm,
                                     is_training=is_training,
                                     data_format=data_format)
            
            flow = predict_flow(inputs,
                                name='flow{}'.format(i),
                                is_training=is_training,
                                data_format=data_format) * 256.

            inputs = tf.concat([inputs, flow], axis=1 if data_format=='channels_first' else -1)
            
            if data_format == 'channels_first':
                flow = tf.transpose(flow, [0,2,3,1])

            flow_dict['flow{}'.format(i)] = flow
    return flow_dict

def model(event_image, is_training=True, data_format=None, do_batch_norm=False):
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    with tf.variable_scope('vs'):
        if data_format == 'channels_first':
            inputs = tf.transpose(event_image, [0,3,1,2])
        else:
            inputs = event_image

        inputs, skip_connections = encoder(inputs, is_training, data_format,
                                           do_batch_norm=do_batch_norm)
        inputs = transition(inputs, is_training, data_format,
                            do_batch_norm=do_batch_norm)
        flow_dict = decoder(inputs, skip_connections, is_training, data_format,
                            do_batch_norm=do_batch_norm)

    return flow_dict
