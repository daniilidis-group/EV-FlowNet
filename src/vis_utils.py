#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
import cv2

"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
"""
def draw_color_wheel_tf(width, height):
    color_wheel_x = tf.lin_space(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = tf.lin_space(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = tf.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_flow = tf.stack([color_wheel_X, color_wheel_Y], axis=2)
    color_wheel_flow = tf.expand_dims(color_wheel_flow, 0)
    color_wheel_rgb, flow_norm, flow_ang = flow_viz_tf(color_wheel_flow)
    return color_wheel_rgb

def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = np.linspace(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_rgb = flow_viz_np(color_wheel_X, color_wheel_Y)
    return color_wheel_rgb

"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
"""
def flow_viz_tf(flow):
    flow_norm = tf.norm(flow, axis=3)
    
    flow_ang_rad = tf.atan2(flow[:, :, :, 1], flow[:, :, :, 0])
    flow_ang = (flow_ang_rad / math.pi) / 2. + 0.5
    
    const_mat = tf.ones(tf.shape(flow_norm))
    hsv = tf.stack([flow_ang, const_mat, flow_norm], axis=3)
    flow_rgb = tf.image.hsv_to_rgb(hsv)
    return flow_rgb, flow_norm, flow_ang_rad

def flow_viz_np(flow_x, flow_y):
    import cv2
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

