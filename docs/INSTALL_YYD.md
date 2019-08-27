#20190827 

#创建虚拟环境

conda remove -n open-mmlab --all　#删除之前的环境
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

#去掉默认路径

echo $PYTHONPATH 　#查看路径，/usr/local/site-packages

unset PYTHONPATH

#安装gcc

which gcc   　#/usr/bin/gcc
module load gcc-5.4.0 #或conda install -c psi4 gcc-5 

which gcc       #/data/nfs_share/gcc-build/gcc-5.4.0/bin/gcc

#安装CUDA9.0版本对应的pytorch

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch


#检查

$ python 
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>> import torch
>>>

#安装mmcv 

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
#安装mmdetection

cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection/
python setup.py develop　#twice

#安装完成

#下载checkpoints

mkdir checkpoints
cd checkpoints/
wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth


#准备数据

mkdir data
cd data/
ln -s /data/nfs_share/public/detection_data/coco/images/ coco
cd ..

vim configs/faster_rcnn_r50_fpn_1x.py #修改文件的data配置

#test

python tools/test.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --show　#报错thread问题，没法cv2显示图片，改用--out
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --out a.pickle#准备数据

#测试完成
