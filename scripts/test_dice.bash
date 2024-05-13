#!/bin/sh

vost_img_path="/home/davide/data/datasets/VOST"
vost_aug_path="/home/davide/data/datasets/VOST_aug"
output_path="/home/davide/data/experiments/coficut/results"
train_batch=64
test_batch=96
workers=32
loss=dice
epochs=300
test_freq=20
test_n_aug_samples=all
checkpoint_path="/home/davide/data/experiments/coficut/lr=0.0001;train_batch=64;dropout=0.1;hidden_units=,;loss=dice/model_state/best_test_coficut_map_m.pth"

python run.py \
../datasets/vost_aug \
 "$vost_img_path" \
 "$vost_aug_path" \
 "$output_path" \
--train_batch $train_batch --train_workers $workers \
--test_batch $test_batch --test_workers $workers --test_n_aug_samples $test_n_aug_samples \
--loss $loss --epochs $epochs --test_frequency $test_freq  \
--coficut_path /home/davide/data/datasets/COFICUT --test_only \
--checkpoint_path "$checkpoint_path"