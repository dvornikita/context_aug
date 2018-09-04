#!/bin/bash

# Download the data.
cd $ROOT/Data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
