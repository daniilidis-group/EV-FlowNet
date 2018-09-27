#!/usr/bin/env python

import math
import os
import argparse

import rospy
from rosbag import Bag
from cv_bridge import CvBridge

import cv2
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _save_events(events,
                 image_times,
                 event_count_images,
                 event_time_images,
                 event_image_times,
                 rows,
                 cols,
                 max_aug,
                 n_skip,
                 event_image_iter, 
                 prefix, 
                 cam,
                 tf_writer,
                 t_start_ros):
    event_iter = 0
    cutoff_event_iter = 0
    image_iter = 0
    curr_image_time = (image_times[image_iter] - t_start_ros).to_sec()
    
    event_count_image = np.zeros((rows, cols, 2), dtype=np.uint16)
    event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)

    while image_iter < len(image_times) and \
          events[-1][2] > curr_image_time:
        x = events[event_iter][0]
        y = events[event_iter][1]
        t = events[event_iter][2]

        if events[event_iter][3] > 0:
            event_count_image[y, x, 0] += 1
            event_time_image[y, x, 0] = t
        else:
            event_count_image[y, x, 1] += 1
            event_time_image[y, x, 1] = t

        event_iter += 1
        if t > curr_image_time:
            event_count_images.append(event_count_image)
            event_count_image = np.zeros((rows, cols, 2), dtype=np.uint16)
            event_time_images.append(event_time_image)
            event_time_image = np.zeros((rows, cols, 2), dtype=np.float32)
            cutoff_event_iter = event_iter
            event_image_times.append(image_times[image_iter].to_sec())
            image_iter += n_skip
            if (image_iter < len(image_times)):
                curr_image_time = (image_times[image_iter] - t_start_ros).to_sec()

    del image_times[:image_iter]
    del events[:cutoff_event_iter]
    
    if len(event_count_images) >= max_aug:
        n_to_save = len(event_count_images) - max_aug + 1
        for i in range(n_to_save):
            image_times_out = np.array(event_image_times[i:i+max_aug+1])
            image_times_out = image_times_out.astype(np.float64)
            event_time_images_np = np.array(event_time_images[i:i+max_aug], dtype=np.float32)
            event_time_images_np -= image_times_out[0] - t_start_ros.to_sec()
            event_time_images_np = np.clip(event_time_images_np, a_min=0, a_max=None)
            image_shape = np.array(event_time_images_np.shape, dtype=np.uint16)
            
            event_image_tf = tf.train.Example(features=tf.train.Features(feature={
                'image_iter': _int64_feature(event_image_iter),
                'shape': _bytes_feature(
                    image_shape.tobytes()),
                'event_count_images': _bytes_feature(
                    np.array(event_count_images[i:i+max_aug], dtype=np.uint16).tobytes()),
                'event_time_images': _bytes_feature(event_time_images_np.tobytes()),
                'image_times': _bytes_feature(
                    image_times_out.tobytes()),
                'prefix': _bytes_feature(
                    prefix.encode()),
                'cam': _bytes_feature(
                    cam.encode())
            }))

            tf_writer.write(event_image_tf.SerializeToString())
            event_image_iter += n_skip

        del event_count_images[:n_to_save]
        del event_time_images[:n_to_save]
        del event_image_times[:n_to_save]
    return event_image_iter

def main():
    parser = argparse.ArgumentParser(
        description=("Extracts grayscale and event images from a ROS bag and "
                     "saves them as TFRecords for training in TensorFlow."))
    parser.add_argument("--bag", dest="bag",
                        help="Path to ROS bag.",
                        required=True)
    parser.add_argument("--prefix", dest="prefix",
                        help="Output file prefix.",
                        required=True)
    parser.add_argument("--output_folder", dest="output_folder",
                        help="Output folder.",
                        required=True)
    parser.add_argument("--max_aug", dest="max_aug",
                        help="Maximum number of images to combine for augmentation.",
                        type=int,
                        default = 6)
    parser.add_argument("--n_skip", dest="n_skip",
                        help="Maximum number of images to combine for augmentation.",
                        type=int,
                        default = 1)
    parser.add_argument("--start_time", dest="start_time",
                        help="Time to start in the bag.",
                        type=float,
                        default = 0.0)
    parser.add_argument("--end_time", dest="end_time",
                        help="Time to end in the bag.",
                        type=float,
                        default = -1.0)

    args = parser.parse_args()
    
    bridge = CvBridge()

    n_msgs = 0
    left_event_image_iter = 0
    right_event_image_iter = 0
    left_image_iter = 0
    right_image_iter = 0
    first_left_image_time = -1
    first_right_image_time = -1

    left_events = []
    right_events = []
    left_images = []
    right_images = []
    left_image_times = []
    right_image_times = []
    left_event_count_images = []
    left_event_time_images = []
    left_event_image_times = []
    
    right_event_count_images = []
    right_event_time_images = []
    right_event_image_times = []
    
    cols = 346
    rows = 260
    print("Processing bag")
    bag = Bag(args.bag)

    left_tf_writer = tf.python_io.TFRecordWriter(
        os.path.join(args.output_folder, args.prefix, "left_event_images.tfrecord"))
    right_tf_writer = tf.python_io.TFRecordWriter(
        os.path.join(args.output_folder, args.prefix, "right_event_images.tfrecord"))
    
    # Get actual time for the start of the bag.
    t_start = bag.get_start_time()
    t_start_ros = rospy.Time(t_start)
    # Set the time at which the bag reading should end.
    if args.end_time == -1.0:
        t_end = bag.get_end_time()
    else:
        t_end = t_start + args.end_time

    for topic, msg, t in bag.read_messages(
            topics=['/davis/left/image_raw',
                    '/davis/right/image_raw',
                    '/davis/left/events',
                    '/davis/right/events'],
            start_time=rospy.Time(args.start_time + t_start),
            end_time=rospy.Time(t_end)):
        # Check to make sure we're working with stereo messages.
        if not ('left' in topic or 'right' in topic):
            print('ERROR: topic {} does not contain left or right, is this stereo?'
                  'If not, you will need to modify the topic names in the code.'.
                  format(topic))
            return
        
        # Counter for status updates.
        n_msgs += 1
        if n_msgs % 500 == 0:
            print("Processed {} msgs, {} images, time is {}.".format(n_msgs,
                                                                    left_event_image_iter,
                                                                    t.to_sec() - t_start))
        
        isLeft = 'left' in topic
        if 'image' in topic:
            width = msg.width
            height = msg.height
            if width != cols or height != rows:
                print("Image dimensions are not what we expected: set: ({} {}) vs  got:({} {})"
                      .format(cols, rows, width, height))
                return
            time = msg.header.stamp
            image = np.asarray(bridge.imgmsg_to_cv2(msg, msg.encoding))
            image = np.reshape(image, (height, width))

            if isLeft:
                cv2.imwrite(os.path.join(args.output_folder,
                                         args.prefix,
                                         "left_image{:05d}.png".format(left_image_iter)),
                            image)
                if left_image_iter > 0:
                    left_image_times.append(time)
                else:
                    first_left_image_time = time
                    left_event_image_times.append(time.to_sec())
                left_image_iter += 1
            else:
                cv2.imwrite(os.path.join(args.output_folder,
                                         args.prefix,
                                         "right_image{:05d}.png".format(right_image_iter)),
                            image)
                if right_image_iter > 0:
                    right_image_times.append(time)
                else:
                    first_right_image_time = time
                    right_event_image_times.append(time.to_sec())
                right_image_iter += 1
        elif 'events' in topic and msg.events:
            # Add events to list.
            for event in msg.events:
                ts = event.ts
                event = [event.x,
                         event.y,
                         (ts - t_start_ros).to_sec(),
                         (float(event.polarity) - 0.5) * 2]
                if isLeft and first_left_image_time != -1  and ts > first_left_image_time:
                    left_events.append(event)
                elif first_right_image_time != -1 and ts > first_right_image_time:
                    right_events.append(event)
            if isLeft:
                if len(left_image_times) >= args.max_aug and\
                   left_events[-1][2] > (left_image_times[args.max_aug-1]-t_start_ros).to_sec():
                    left_event_image_iter = _save_events(left_events,
                                                         left_image_times,
                                                         left_event_count_images,
                                                         left_event_time_images,
                                                         left_event_image_times,
                                                         rows,
                                                         cols,
                                                         args.max_aug,
                                                         args.n_skip,
                                                         left_event_image_iter, 
                                                         args.prefix, 
                                                         'left',
                                                         left_tf_writer,
                                                         t_start_ros)
            else:
                if len(right_image_times) >= args.max_aug and\
                   right_events[-1][2] > (right_image_times[args.max_aug-1]-t_start_ros).to_sec():
                    right_event_image_iter = _save_events(right_events,
                                                          right_image_times,
                                                          right_event_count_images,
                                                          right_event_time_images,
                                                          right_event_image_times,
                                                          rows,
                                                          cols,
                                                          args.max_aug,
                                                          args.n_skip,
                                                          right_event_image_iter, 
                                                          args.prefix, 
                                                          'right',
                                                          right_tf_writer,
                                                          t_start_ros)

    left_tf_writer.close()
    right_tf_writer.close()
    
    image_counter_file = open(os.path.join(args.output_folder, args.prefix, "n_images.txt") , 'w')
    image_counter_file.write("{} {}".format(left_event_image_iter, right_event_image_iter))
    image_counter_file.close()            
    
if __name__ == "__main__":
    main()
