#!/bin/bash
DATASET_FILENAME="AmazonCat-13K_tfidf_train_ver1.svm.bz2"
DATASET_PATH="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/$DATASET_FILENAME"
PYTHON_INTERPRETER="python3"

if not test -e "data/$DATASET_FILENAME"; then
    echo "File $DATASET_FILENAME does not exist. Downloading"
    wget -P data $DATASET_PATH
else
    echo "File exists, preprocessing..."
    $PYTHON_INTERPRETER scripts/prepare_data.py
fi