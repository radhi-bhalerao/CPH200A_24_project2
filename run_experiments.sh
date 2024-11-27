#!/bin/bash

# run grid search for part 1.1
python scripts/dispatcher1.py

# run sweep for part 1.1.
python scripts/sweep.py

# train risk model (with resnet3d backbone) for part 3.2
python scripts/main.py --risk --model resnet3d --train True --dataset_name nlst --batch_size 5 --pretraining True --class_balance True --max_followup 6