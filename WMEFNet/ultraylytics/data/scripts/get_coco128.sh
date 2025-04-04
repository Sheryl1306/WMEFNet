#!/bin/bash

# Download/unzip images and labels
d='../datasets' # unzip directory
url=https://github.com/ultralytics/detr/releases/download/v1.0/
f='coco128.zip' # or 'coco128-segments.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

wait # finish background tasks
