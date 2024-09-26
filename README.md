# Channel-wise Contribution Assessment for RGB-D Salient Object Detection

## Requirements
- python 3.8
- cuda 11.6.2
- pytorch
- Opencv python
## Main-step
#### 1.Channel-wise Early Fusion: 
Run channel_wise_fusion.py
#### 2.SLIC to Fine-grained Channel Contribution Assessment: 
Run SLIC.py
#### 3.Saliency Sensing Tool: 
As shown in Figure Method pipeline, we use the [EDN](https://arxiv.org/pdf/2012.13093). Please run "SOD_tool/EDN/edn_test.py"
#### 4.Contribution-aware Fusion: 
We use [BBRF](https://ieeexplore.ieee.org/abstract/document/10006743) and [CAVER](https://ieeexplore.ieee.org/abstract/document/10015667). Firstly, run  "SOD_tool/BBRF/test.py" and "SOD_tool/CAVER/main.py". Secondly, run fusion.py to get final saliency maps.

## Datasets: 
The document contains nine available RGB-D SOD datasets: NJU2K, NLPR, SIP, STERE, SSD, LFSD, DUT, ReDWeb-S and COME15K-E. 
- [Baidu Pan link] (https://pan.baidu.com/s/1YK_UmDA3J8jmDxT9AXKIvQ), with the code: 41eh
## Results
- Coarse-grained saliency maps - [Baidu Pan link] (https://pan.baidu.com/s/1rIdL1Om0XAICeD4Z66ic-A), with the code: j5z2
- Fine-grained saliency maps - [Baidu Pan link] (https://pan.baidu.com/s/1XARw0nSlj-M2OWuKO6ZG7A), with the code: drxa
## Acknowledgement
Thanks to [EDN](https://arxiv.org/pdf/2012.13093), [BBRF](https://ieeexplore.ieee.org/abstract/document/10006743) and [CAVER](https://ieeexplore.ieee.org/abstract/document/10015667).


