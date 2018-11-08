#!/bin/bash

num_indoor_flying_bags=4

bagnames[1]=sep5_4_data
bagnames[2]=sep5_5_data
bagnames[3]=sep5_6_data
bagnames[4]=sep5_7_data

starttimes[1]=4.0
starttimes[2]=9.0
starttimes[3]=7.0
starttimes[4]=6.0

output_folder=mvsec_aug
max_aug=6
flying_bag_dir=mvsec_bags/indoor_flying/

for bag_iter in `seq 1 $num_indoor_flying_bags`;
do
    echo Extracting bag$bag_iter
    mkdir ${output_folder}/indoor_flying$bag_iter
    python extract_rosbag_to_tf.py --bag $flying_bag_dir${bagnames[$bag_iter]}.bag --prefix indoor_flying$bag_iter --start_time ${starttimes[$bag_iter]} --max_augmentation $max_aug --n_skip 1 --output_folder $output_folder
done

outdoor_day_bag_dir=mvsec_bags/outdoor_day/
num_outdoor_day_bags=2

outdoor_day_bagnames[1]=sep22_04_data
outdoor_day_bagnames[2]=sep22_36_data

outdoor_day_starttimes[1]=3.0
outdoor_day_starttimes[2]=45.0

for bag_iter in `seq 1 $num_outdoor_day_bags`;
do
    echo Extracting driving day bag$bag_iter
    mkdir ${output_folder}/outdoor_day$bag_iter
    python extract_rosbag_to_tf.py --bag $outdoor_day_bag_dir${outdoor_day_bagnames[$bag_iter]}.bag --prefix outdoor_day$bag_iter --start_time ${outdoor_day_starttimes[$bag_iter]} --max_augmentation ${max_aug} --n_skip 1 --output_folder $output_folder
done

outdoor_night_bag_dir=mvsec_bags/outdoor_night/
num_outdoor_night_bags=3

outdoor_night_bagnames[1]=sep8_09_data
outdoor_night_bagnames[2]=sep8_14_data
outdoor_night_bagnames[3]=sep8_23_data

outdoor_night_starttimes[1]=0.0
outdoor_night_starttimes[2]=0.0
outdoor_night_starttimes[3]=0.0

for bag_iter in `seq 1 $num_outdoor_night_bags`;
do
    echo Extracting driving night bag$bag_iter
    mkdir ${output_folder}/outdoor_night$bag_iter
    python extract_rosbag_to_tf.py --bag $outdoor_night_bag_dir${outdoor_night_bagnames[$bag_iter]}.bag --prefix outdoor_night$bag_iter --start_time ${outdoor_night_starttimes[$bag_iter]} --max_augmentation ${max_aug} --n_skip 1 --output_folder $output_folder
done
