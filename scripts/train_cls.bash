#!/bin/sh

vost_img_path="/home/davide/data/datasets/VOST"
vost_aug_path="/home/davide/data/datasets/VOST_aug"
output_path="/home/davide/data/experiments/coficut"
train_batch=64
test_batch=96
workers=32
loss=bce
epochs=300
test_freq=20
test_n_aug_samples=5  # keep this low, then test all using the dedicated script

python run.py \
../datasets/vost_aug \
 "$vost_img_path" \
 "$vost_aug_path" \
 "$output_path" \
--train_batch $train_batch --train_workers $workers \
--test_batch $test_batch --test_workers $workers --test_n_aug_samples $test_n_aug_samples \
--loss $loss --epochs $epochs --test_frequency $test_freq  \
--coficut_path /home/davide/data/datasets/COFICUT \
--split_vost_by 'change_ratio' --run_tags split_vost_by