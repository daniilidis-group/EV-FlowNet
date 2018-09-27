#!/usr/bin/env python
import tensorflow as tf

"""
Generates a prediction of an image given the optical flow, as in Spatial Transformer Networks.
"""
def warp_images_with_flow(images, flow):
    height  = tf.shape(flow)[1]
    width = tf.shape(flow)[2]
    flow_x,flow_y = tf.split(flow,[1,1],axis = 3)
    coord_x, coord_y = tf.meshgrid(tf.range(width), tf.range(height))

    pos_x = tf.to_float(tf.expand_dims(tf.expand_dims(coord_x, axis = 2), axis=0)) +\
            flow_x
    pos_y = tf.to_float(tf.expand_dims(tf.expand_dims(coord_y, axis = 2), axis=0)) +\
            flow_y
    warped_points = tf.concat([pos_x, pos_y], axis = 3, name = 'warped_points')
    return tf.contrib.resampler.resampler(images, warped_points)

"""
Robust Charbonnier loss, as defined in equation (4) of the paper.
"""
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = tf.reduce_mean(tf.pow(tf.square(delta)+tf.square(epsilon), alpha))
    return loss

"""
Local smoothness loss, as defined in equation (5) of the paper.
The neighborhood here is defined as the 8-connected region around each pixel.
"""
def compute_smoothness_loss(flow):
    flow_ucrop = flow[..., 1:, :]
    flow_dcrop = flow[..., :-1, :]
    flow_lcrop = flow[:, 1:, ...]
    flow_rcrop = flow[:, :-1, ...]

    flow_ulcrop = flow[:, 1:, 1:, :]
    flow_drcrop = flow[:, :-1, :-1, :]
    flow_dlcrop = flow[:, :-1, 1:, :]
    flow_urcrop = flow[:, 1:, :-1, :]
    
    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) +\
                      charbonnier_loss(flow_ucrop - flow_dcrop) +\
                      charbonnier_loss(flow_ulcrop - flow_drcrop) +\
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /= 4.
    
    return smoothness_loss

"""
Multi-scale photometric loss, as defined in equation (3) of the paper.
"""
def compute_photometric_loss(prev_images, next_images, event_images, flow_dict):
    total_photometric_loss = 0.
    loss_weight_sum = 0.

    for i in range(len(flow_dict)):
        flow = flow_dict["flow{}".format(i)]
        height = tf.shape(flow)[1]
        width = tf.shape(flow)[2]
        
        prev_images_resize = tf.image.resize_images(prev_images,
                                                    [height, width],
                                                    align_corners=True)
        next_images_resize = tf.image.resize_images(next_images,
                                                    [height, width],
                                                    align_corners=True)
        
        next_images_warped = warp_images_with_flow(next_images_resize, flow)

        photometric_loss = charbonnier_loss(next_images_warped - prev_images_resize)
        total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum

    return total_photometric_loss
