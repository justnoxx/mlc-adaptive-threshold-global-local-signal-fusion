# mlc-adaptive-threshold-global-local-signal-fusion

## Package structure
* scripts: helper scrips
* mlc_adaptive_threshold: source code
* notebooks: useful notebooks for visualisation.

## How to run

1. Download dataset.
2. Build required .pt tensors.
3. Install this package
4. Run training script.


Download dataset from:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html

The file we need is: AmazonCat-13K_tfidf_train_ver1.svm.bz2

Direct link:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/AmazonCat-13K_tfidf_train_ver1.svm.bz2

OR

Run script:
`chmod +x scripts/download_and_prepare_dataset.sh`

And from the current repo folder:
`./scripts/download_and_prepare_dataset.sh`