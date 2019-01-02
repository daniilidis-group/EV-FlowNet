#!/usr/bin/env python
import os
import time

import tensorflow as tf
import numpy as np

from config import *
from data_loader import get_loader
from eval_utils import *
from model import *
from vis_utils import *

def drawImageTitle(img, title):
    cv2.putText(img,
                title,
                (60, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=2,
                bottomLeftOrigin=False)
    return img

def test(sess,
         args,
         event_image_loader,
         prev_image_loader,
         next_image_loader,
         timestamp_loader):
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope('vs'):
        flow_dict = model(event_image_loader,
                          is_training=False,
                          do_batch_norm=not args.no_batch_norm)
    
    event_image = tf.reduce_sum(event_image_loader[:, :, :, :2], axis=-1, keepdims=True)
    flow_rgb, flow_norm, flow_ang_rad = flow_viz_tf(flow_dict['flow3'])
    color_wheel_rgb = draw_color_wheel_np(args.image_width, args.image_height)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    saver = tf.train.Saver()
    saver.restore(sess, args.load_path)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    max_flow_sum = 0
    min_flow_sum = 0
    iters = 0
    
    if args.test_plot:
        import cv2
        cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)

    if args.gt_path:
        print("Loading ground truth {}".format(args.gt_path))
        gt = np.load(args.gt_path)
        gt_timestamps = gt['timestamps']
        U_gt_all = gt['x_flow_dist']
        V_gt_all = gt['y_flow_dist']
        print("Ground truth loaded")
    
        AEE_sum = 0.
        percent_AEE_sum = 0.
        AEE_list = []

    if args.save_test_output:
        output_flow_list = []
        gt_flow_list = []
        event_image_list = []
        
    while not coord.should_stop():
        start_time = time.time()
        try:
            flow_dict_np,\
                prev_image,\
                next_image,\
                event_image,\
                image_timestamps = sess.run([flow_dict,
                                             prev_image_loader,
                                             next_image_loader,
                                             event_image_loader,
                                             timestamp_loader])
        except tf.errors.OutOfRangeError:
            break
        
        network_duration = time.time() - start_time
        event_image = np.array(event_image)
        
        pred_flow = np.squeeze(flow_dict_np['flow3'])

        max_flow_sum += np.max(pred_flow)
        min_flow_sum += np.min(pred_flow)
        
        event_count_image = np.sum(event_image[..., :2], axis=-1)
        event_count_image = (event_count_image * 255 / event_count_image.max()).astype(np.uint8)
        event_count_image = np.squeeze(event_count_image)

        if args.save_test_output:
            output_flow_list.append(pred_flow)
            event_image_list.append(event_count_image)
        
        if args.gt_path:
            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all,
                                                        gt_timestamps,
                                                        image_timestamps[0][0],
                                                        image_timestamps[0][1])
            
            gt_flow = np.stack((U_gt, V_gt), axis=2)

            if args.save_test_output:
                gt_flow_list.append(gt_flow)
            
            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2
            
            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]       
        
            # Calculate flow error.
            AEE, percent_AEE, n_points = flow_error_dense(gt_flow, 
                                                          pred_flow, 
                                                          event_count_image,
                                                          'outdoor' in args.test_sequence)
            AEE_list.append(AEE)
            AEE_sum += AEE
            percent_AEE_sum += percent_AEE
            
        iters += 1
        if iters % 100 == 0:
            print('-------------------------------------------------------')
            print('Iter: {}, time: {:f}, run time: {:.3f}s\n'
                  'Mean max flow: {:.2f}, mean min flow: {:.2f}'
                  .format(iters, image_timestamps[0][0], network_duration,
                          max_flow_sum / iters, min_flow_sum / iters))
            if args.gt_path:
                print('Mean AEE: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
                      .format(AEE_sum / iters,
                              percent_AEE_sum / iters,
                              n_points))

        # Prep outputs for nice visualization.
        if args.test_plot:
            pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
            pred_flow_rgb = drawImageTitle(pred_flow_rgb, 'Predicted Flow')
            
            event_time_image = np.squeeze(np.amax(event_image[..., 2:], axis=-1))
            event_time_image = (event_time_image * 255 / event_time_image.max()).astype(np.uint8)
            event_time_image = np.tile(event_time_image[..., np.newaxis], [1, 1, 3])
            
            event_count_image = np.tile(event_count_image[..., np.newaxis], [1, 1, 3])

            event_time_image = drawImageTitle(event_time_image, 'Timestamp Image')
            event_count_image = drawImageTitle(event_count_image, 'Count Image')
            
            prev_image = np.squeeze(prev_image)
            prev_image = np.tile(prev_image[..., np.newaxis], [1, 1, 3])

            prev_image = drawImageTitle(prev_image, 'Grayscale Image')
            
            gt_flow_rgb = np.zeros(pred_flow_rgb.shape)
            errors = np.zeros(pred_flow_rgb.shape)

            gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow - No GT')
            errors = drawImageTitle(errors, 'Flow Error - No GT')
            
            if args.gt_path:
                errors = np.linalg.norm(gt_flow - pred_flow, axis=-1)
                errors = (errors * 255. / errors.max()).astype(np.uint8)
                errors = np.tile(errors[..., np.newaxis], [1, 1, 3])
                errors[event_count_image == 0] = 0

                if 'outdoor' in args.test_sequence:
                    errors[190:, :] = 0
                
                gt_flow_rgb = flow_viz_np(gt_flow[...,0], gt_flow[...,1])

                gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow')
                errors= drawImageTitle(errors, 'Flow Error')
                
            top_cat = np.concatenate([event_count_image, prev_image, pred_flow_rgb], axis=1)
            bottom_cat = np.concatenate([event_time_image, errors, gt_flow_rgb], axis=1)
            cat = np.concatenate([top_cat, bottom_cat], axis=0)
            cat = cat.astype(np.uint8)
            cv2.imshow('EV-FlowNet Results', cat)
            cv2.waitKey(1)
            
    print('Testing done. ')
    if args.gt_path:
        print('mean AEE {:02f}, mean %AEE {:02f}'
              .format(AEE_sum / iters, 
                      percent_AEE_sum / iters))
    if args.save_test_output:
        if args.gt_path:
            print('Saving data to {}_output_gt.npz'.format(args.test_sequence))
            np.savez('{}_output_gt.npz'.format(args.test_sequence),
                     output_flows=np.stack(output_flow_list, axis=0),
                     gt_flows=np.stack(gt_flow_list, axis=0),
                     event_images=np.stack(event_image_list, axis=0))
        else:
            print('Saving data to {}_output.npz'.format(args.test_sequence))
            np.savez('{}_output.npz'.format(args.test_sequence),
                     output_flows=np.stack(output_flow_list, axis=0),
                     event_images=np.stack(event_image_list, axis=0))

    coord.request_stop()

def main():        
    args = configs()
    args.load_path = tf.train.latest_checkpoint(os.path.join(args.load_path,
                                                             args.training_instance))

    sess = tf.Session()
    event_image_loader, prev_image_loader, next_image_loader, timestamp_loader, n_ima = get_loader(
        args.data_path,
        1,
        args.image_width,
        args.image_height,
        split='test',
        shuffle=False,
        sequence=args.test_sequence,
        skip_frames=args.test_skip_frames,
        time_only=args.time_only,
        count_only=args.count_only)

    if not args.load_path:
        raise Exception("You need to set `load_path` and `training_instance`.")
            
    print("Read {} images".format(n_ima))
    test(sess,
         args,
         event_image_loader,
         prev_image_loader,
         next_image_loader,
         timestamp_loader)
    sess.close()


if __name__ == "__main__":
    main()
