# EfficientNet-Light-head-RCNN for Human Detection


Google has recently released new efficient model architectures for edge devices, I thought it would be cool to see what the results of these models are as a backbone for FasterRCNNN.
The EfficientNet-Light-head-RCNN has 31MAP on pedestrian detection and run at 12 FPS on my GTX 1060.
Only EfficientNet B0 was tested.

### Downloading the data

the crowdHuman Dataset can be downloaed here [CrowdHuman](https://www.crowdhuman.org/)

### Data folder Structure (please create a data folder and make it like this)
    ==>data
    ===>annotations
    ===>Images
    ===>Images_test
    ====>Images_validation
    
### COCO Evaluation Results

| Average Precision  (AP)  |IoU=0.50:0.95   | area=   all  |maxDets=100    | 0.310  |
|---|---|---|---|---|
| Average Precision  (AP)  | IoU=0.50   | area=   all |maxDets=100   |0.589  |
| Average Precision  (AP)   | IoU=0.75   | area=   all  |maxDets=100   | 0.295  |


### demo

![Alt text](demo.jpg?raw=true "Person detection")


### Pretrained Model

if you want to use the pretrained model, please download, create a folder named checkpoint and put it inside
[Pretrained_model](https://www.dropbox.com/s/hyc453tmlskz8of/efficient_model_L_7.pth?dl=0)



### Training (once you have downlaoded the dataset)

python3 Train.py


### Citing

```
@article{li2017light,
  title={Light-Head R-CNN: In Defense of Two-Stage Object Detector},
  author={Li, Zeming and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Deng, Yangdong and Sun, Jian},
  journal={arXiv preprint arXiv:1711.07264},
  year={2017}
}
```


```
@article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
}
```




