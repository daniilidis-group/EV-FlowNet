#!/usr/bin/env python
import math
import tensorflow as tf
import numpy as np

from losses import *
from model import model
from vis_utils import *

class EVFlowNet():
    def __init__(self,
                 args,
                 event_img_loader,
                 prev_img_loader,
                 next_img_loader,
                 n_ima,
                 is_training = True,
                 weight_decay_weight = 1e-4):
        self._args = args
        
        self._event_img_loader = event_img_loader
        self._prev_img_loader = prev_img_loader
        self._next_img_loader = next_img_loader

        self._n_ima = n_ima
        self._weight_decay_weight = weight_decay_weight
        self._is_training = is_training
        
    def _build_graph(self, global_step):
        #Model
        with tf.variable_scope('vs'):
            flow_dict = model(self._event_img_loader, 
                              self._is_training,
                              do_batch_norm=not self._args.no_batch_norm)
        
        with tf.name_scope('loss'):
            # Weight decay loss.
            with tf.name_scope('weight_decay'):
                var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vs/')
                for v in var:
                    wd_loss = tf.multiply(tf.nn.l2_loss(v), self._weight_decay_weight)
                    tf.add_to_collection('wd_loss', wd_loss)
                weight_decay_loss = tf.add_n(tf.get_collection('wd_loss'))
                tf.summary.scalar('weight_decay_loss', weight_decay_loss)

            # Smoothness loss.
            smoothness_loss = 0
            for i in range(len(flow_dict)):
                smoothness_loss += compute_smoothness_loss(flow_dict["flow{}".format(i)])
            smoothness_loss *= self._args.smoothness_weight / 4.
            tf.summary.scalar("smoothness_loss", smoothness_loss)

            # Photometric loss.
            photometric_loss = compute_photometric_loss(self._prev_img_loader, 
                                                        self._next_img_loader,
                                                        self._event_img_loader,
                                                        flow_dict)
            tf.summary.scalar('photometric_loss', photometric_loss)

            # Warped next image for debugging.
            next_image_warped = warp_images_with_flow(self._next_img_loader,
                                                      flow_dict['flow3'])
            
            loss = weight_decay_loss + photometric_loss + smoothness_loss
            tf.summary.scalar('total_loss', loss)
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            learning_rate = tf.train.exponential_decay(self._args.initial_learning_rate, 
                                                       global_step,
                                                       4. * self._n_ima / self._args.batch_size, 
                                                       self._args.learning_rate_decay,
                                                       staircase=True)
            tf.summary.scalar('lr', learning_rate)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
        return flow_dict, loss, optimizer, next_image_warped

    def train(self):
        print("Starting training.")
        global_step = tf.train.get_or_create_global_step()

        #load data
        flow_dict, self._loss, self._optimizer, next_image_warped = self._build_graph(global_step)

        final_flow = flow_dict['flow3']
        
        train_op = tf.contrib.slim.learning.create_train_op(total_loss=self._loss,
                                                            optimizer=self._optimizer,
                                                            global_step=global_step)
        # Visualization for Tensorboard.
        with tf.device('/cpu:0'):
            event_img = tf.expand_dims(self._event_img_loader[:, :, :, 0] +\
                                       self._event_img_loader[:, :, :, 1], axis=3)
            
            event_time_img = tf.reduce_max(self._event_img_loader[:, :, :, 2:4],
                                           axis=3,
                                           keepdims=True)
            flow_rgb, flow_norm, flow_ang_rad = flow_viz_tf(final_flow)

            image_error = tf.abs(next_image_warped - self._prev_img_loader)
            image_error = tf.clip_by_value(image_error, 0., 20.)

            # Color wheel to visualize the flow directions.
            color_wheel_rgb = draw_color_wheel_tf(self._args.image_width, self._args.image_height)

            # Appending letters to each title allows us to control the order of display.
            tf.summary.image("a-Color wheel",
                             color_wheel_rgb,
                             max_outputs=1)
            tf.summary.image("b-Flow",
                             flow_rgb,
                             max_outputs=self._args.batch_size)
            tf.summary.image("c-Event time image", event_time_img,
                             max_outputs=self._args.batch_size)
            tf.summary.image('d-Warped_next_image', next_image_warped,
                             max_outputs=self._args.batch_size)
            tf.summary.image("e-Prev image",
                             self._prev_img_loader,
                             max_outputs=self._args.batch_size)
            tf.summary.image("f-Image error",
                             image_error,
                             max_outputs=self._args.batch_size)
            tf.summary.image("g-Event image",
                             event_img,
                             max_outputs=self._args.batch_size)
            
        writer = tf.summary.FileWriter(self._args.summary_path)

        debug_rate = 5000
        tf.logging.set_verbosity(tf.logging.DEBUG)

        session_config = tf.ConfigProto()
        
        tf.contrib.slim.learning.train(
            train_op=train_op,
            logdir=self._args.load_path,
            number_of_steps=600000,
            log_every_n_steps=debug_rate,
            save_summaries_secs=240.,
            summary_writer=writer,
            save_interval_secs=240.)
            
if __name__ == "__main__":
    main()
