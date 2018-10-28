# GRCNN.pytorch
Gated Recurrent Convolution Neural Network in Pytorch. <br>
This is a pytorch implementation of [Gated Recurrent Convolution Neural Network for OCR](http://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
## Requirements
1. Pytorch >= 0.4.0 <br>
2. opencv <br>
3. [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc)
4. tqdm
## Construct dataset
The names of train and val data are construct bellow:
```
img_xxx_$$$.jpg
```
where 'xxx' represents the number of this image and '$$$' represents the label of this image. <br>
For examples, img_0_WHLU.jpg <br>
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
## Notes
