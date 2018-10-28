# GRCNN.pytorch
Gated Recurrent Convolution Neural Network in Pytorch. <br>
This is a pytorch implementation of [Gated Recurrent Convolution Neural Network for OCR](http://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf).
## Requirements
1. Pytorch >= 0.4.0 <br>
2. opencv <br>
3. [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc)
4. tqdm
## Construct dataset
The names of train and val data are constructed as bellow:
```
img_xxx_$$$.jpg
```
where 'xxx' represents the number of this image and '$$$' represents the label of this image. For example，‘img_0_WHLU.jpg’. <br>
<div align=center><img width="100" height="32" src="https://github.com/jingjing-you/GRCNN.pytorch/blob/master/data_sample/img_0_WHLU.jpg"/></div>
Other examples are in 
```
data_sample/
```
directory.

## Training the model 
You can run
```
python train.py
```
to train your model.
## References
1. [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch) <br>
2. [https://github.com/Jianfeng1991/GRCNN-for-OCR](https://github.com/Jianfeng1991/GRCNN-for-OCR) <br>
