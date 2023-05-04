#!/bin/bash

FILENAME="/home/saad/dev/crcfp/models/backbone/pretrained/resnet101-5d3b4d8f.pth"

mkdir -p /home/saad/dev/cac_segmentation/models/backbone/pretrained
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O $FILENAME
