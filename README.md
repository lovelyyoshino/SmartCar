# SmartCar（SSD-mobilenetv2）

- this repo is forked from https://github.com/amdegroot/ssd.pytorch. Implemented by pytorch.

<!--| Linux                | Windows and MacOS|
|-------------------------|------------------|
[![Build Status](https://travis-ci.org/lovelyyoshino/SmartCar.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-2)        |    [![Build Status](https://dev.azure.com/yekongxiaogang/QPanda2/_apis/build/status/OriginQ.QPanda-2?branchName=master)](https://dev.azure.com/yekongxiaogang/QPanda2/_build/latest?definitionId=4&branchName=master) -->

Contributions:
1. 基于SSD大框架加入mobilenetv2以实现在嵌入式GPU开发板上的视频流识别
2. 增加 focal loss. 
3. 增加detection.py来单纯验证SSD-mobilenetv2算法
4. 提供rs.py来验证D415传感器性能
5. 在detection_D415.py中，加入D415双目摄像头来实现深度预估，以实现物体的动态抓取
6. 提供 visdom可视化，使用python -m visdom.server即可观察收敛


result(train on voc 2007trainval + 2012, test on voc 2007test):
1. ssd-mobielnetv2 (this repo): 70.27%. (without focal loss).
2. ssd-mobielentv1: 68.% (without COCO pretaining), 72.7% (with COCO pretraining)   https://github.com/chuanqi305/MobileNet-SSD.
3. ssd-vgg16 (paper): 77.20%. 

pretrained model and trained model: 

1. 百度网盘: https://pan.baidu.com/s/1RmOPF4jQYpYlE_8E4DifeQ 提取码: f53n 
2. Google drive: https://drive.google.com/drive/folders/1JoDYukyWZZ-iWVWPhUDD998cSPB3LeUw?usp=sharing

