# Openpose ADDA

## Table of Contents

* [Prerequisites](#prerequisites)
* [Training](#training)
* [Validation](#validation)
* [Pre-trained model](#pre-trained-model)
* [Python demo](#python-demo)
* [Citation](#citation)

## Prerequisites

2. Install requirements `pip install -r requirements.txt`

## Training
Train ADDA
```
python3 train_adda.py
```
* Any configuration about training in config.py
    * Include source folder path and target folder path

## Validation

1. Run `python val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>`

## Pre-trained model <a name="pre-trained-model"/>

The model expects normalized image (mean=[128, 128, 128], scale=[1/256, 1/256, 1/256]) in planar BGR format.
Pre-trained on COCO model is available at: https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth, it has 40% of AP on COCO validation set (38.6% of AP on the val *subset*).

## Python Demo <a name="python-demo"/>

We provide python demo just for the quick results preview. Please, consider c++ demo for the best performance. To run the python demo from a webcam:
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0`

## Citation:

If this helps your research, please cite the paper:

```
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```
