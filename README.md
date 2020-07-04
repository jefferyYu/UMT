# Unified Multimodal Transformer (UMT) for Multimodal Named Entity Recognition (MNER)
Two MNER Datasets and Codes for our ACL'2020 paper: Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer.

Author

Jianfei Yu

jfyu@njust.edu.cn

July 1, 2020

## Data
- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
- Step 2: Change the image path in line 552 and line 554 of the "run_mtmner_crf.py" file
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- Setp 4: Put the pre-trained ResNet-152 model under the folder named "resnet"

## Requirement

* PyTorch 1.0.0
* Python 3.7

## Code Usage

### Training for UMT
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES=2" based on your available GPUs.

```sh
sh run_mtmner_crf.sh
```

- We show our running logs on twitter-2015 and twitter-2017 in the folder "log files". Note that the results are a little bit lower than the results reported in our paper, since the experiments were run on different servers.


## Acknowledgements
- Using these two datasets means you have read and accepted the copyrights set by Twitter and dataset providers.
