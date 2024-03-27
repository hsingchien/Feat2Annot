#!/bin/bash

if [ "$1" = "train-cpu" ]; then
	python run.py train --path=./data --window-size=15 --feature-num=200 --lr=5e-4 --patience=5 --valid-niter=200 --batch-size=256 --dropout=.3
elif [ "$1" = "test-cpu" ]; then
    python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt
elif [ "$1" = "train-gpu" ]; then
	python run.py train --path=./data --window-size=30 --feature-num=200 --lr=5e-4 --patience=5 --valid-niter=200 --batch-size=256 --dropout=.3 --cuda
else
	echo "Invalid Option Selected"
fi
