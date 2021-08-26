# Accurate Camouflaged Object Detection via Mixture Convolution and Interactive Fusion.

This repository is for our paper ["Accurate Camouflaged Object Detection via Mixture Convolution and Interactive Fusion"](hxxx)



## 1. Introduction

The arXiv version of **MCIF-Net** is available at [link](hxxx).

Camouflaged object detection (COD), which aims to identify the objects that conceal themselves into the surroundings, has recently drawn increasing research efforts in the field of computer vision. In practice, the success of deep learning based COD is mainly determined by two key factors, including (i) A significantly large receptive field, which provides rich context information, and (ii) An effective fusion strategy, which aggregates the rich multi-level features for accurate COD. Motivated by these observations, in this paper, we propose a novel deep learning based COD approach, which integrates the large receptive field and effective feature fusion into a unified framework. Specifically, we first extract multi-level features from a backbone network. The resulting features are then fed to the proposed dual-branch mixture convolution modules, each of which utilizes multiple asymmetric convolutional layers and two dilated convolutional layers to extract rich context features from a large receptive field. Finally, we fuse the features using specially-designed multi-level interactive fusion modules, each of which employs an attention mechanism along with feature interaction for effective feature fusion.

The proposed model, called **MCIF-Net** , achieves superior performance on COD task (`0.733 F-measure` and `0.785 S-measure` on CAMO; `0.859 E-measure` and `0.045 MAE`  on COD10K), surpassing cutting-edge models by a large margin.

## 2. Framework overview

![](https://github.com/dongbo811/MCIFNet/blob/main/Figs/net.png)


## 3. Results

### 3.1 Quantitative comparison

![](https://github.com/dongbo811/MCIFNet/blob/main/Figs/visual2.png)
### 3.2 PR and F-measure curves

![](https://github.com/dongbo811/MCIFNet/blob/main/Figs/pr_curve.png)

## 4. Usage:

### 4.1 Recommended environment

```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```

### 4.2 Data preparation

Please download training and testing datasets and move them into ./dataset/, which can be found via [TrainDataset](https://drive.google.com/u/0/uc?id=120wKRvwXpqqeEejw60lYsEyZ4SOicR3M&export=download)/[TestDataset](https://drive.google.com/u/0/uc?id=1bTIb2qo7WXfyLgCn43Pz0ZDQ4XceO9dE&export=download). Please also modify the path in train.py and test.py accordingly.


### 4.3 Pretrained model

You can download the pretrained model (i.e., ResNet-50) from [Baidu Drive](https://pan.baidu.com/s/17o9ixUYJlE_Xr6fjzqUF0Q) [code:yf4k], and then put it in the folder './out' for initialization. 

### 4.4 Training

```
git https://github.com/dongbo811/MCIFNet.git
cd MCIFNet 
python train.py
```

### 4.5 Testing

```
cd MCIFNet 
python test.py
```

### 4.6 Evaluating your trained model

Matlab: Please refer to the work in ([link](https://github.com/DengPingFan/SINet)).

Python: Please refer to the work in ([link](https://github.com/zyjwuyan/SOD_Evaluation_Metrics)).

Note that we use the Python version for evaluation in our work.


### 4.7 Our trained model and results

You can download the trained model from [Baidu Drive](https://pan.baidu.com/s/1logoYpfwNWDawOGotTSnAw) [code:bmw4] and put the model in directory './MCIFNet'. Our results are publicly available at [Baidu Drive](https://pan.baidu.com/s/1RKPgQr-9q81auZNR9_tlCQ) [code:ncte].

## 5. Citation

```
@article{dong2021towards,
  title={Towards accurate camouflaged object detection with mixture convolution and interactive fusion},
  author={Dong, Bo and Zhuge, Mingchen and Wang, Yongxiong and Bi, Hongbo and Chen, Geng},
  journal={arXiv preprint arXiv:2101.05687},
  year={2021}
}
```

## 6. Acknowledgment

We are very grateful for the excellent work made by [F3Net](https://github.com/weijun88/F3Net), which provides the basis for our framework.

## 7. FAQ

If you want to improve the usability or have any piece of advice, please feel free to contact me directly (bodong.cv@gmail.com).

## 8. License

The source code is free for research and education use only. Any commercial use should get formal permission first.

