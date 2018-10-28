#coding:utf-8
import torch.utils.data as data
import torch
import numpy as np
from glob import glob
import cv2
import torchvision.transforms as transforms
import random
import os

class MyDataSet(data.Dataset):
    def __init__(self,floderPath,width,height,max_len):
        self.floderPath=floderPath
        self.dataFileList=glob(self.floderPath+'*.jpg')
        self.len=len(self.dataFileList)
        self.width=width
        self.height=height
        self.alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
        self.alphabet_dict = {}
        self.max_len = max_len
        for ii in range(len(self.alphabet)):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.alphabet_dict[self.alphabet[ii]] = ii + 1

        print('find ',len(self.dataFileList),' images')

        mean=[x/255 for x in [125.3,123.0,113.0]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.dataFileList)

    def __getitem__(self,index):
        try:
            imgPath=self.dataFileList[index]
            img = cv2.imread(imgPath)

            if random.randint(0,10)>7:
                img = self.addSaltNoise(img)
            if random.randint(0,10)>7:
                img = self.addBlurNoise(img)
            img = self.random_crop(img)
               
            txt_name = os.path.basename(imgPath).replace('.jpg', '').split('_')[-1]
            txt_len = len(txt_name)

            #get one hot encode
            txt_label = [0 for ii in range(self.max_len)]
            for ii in range(txt_len):
                txt_label[ii] = self.alphabet_dict[txt_name[ii]]

            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            img = cv2.resize(img, (self.width, self.height))
            img = np.reshape(img, newshape=[1, self.height, self.width])

            img = img.astype(np.float32)
            img = img / 255
            img = img - 0.5
            img = img * 2
        
            img_tensor = torch.from_numpy(img).float()

            txt_len = (torch.zeros(1) + txt_len).int()
            txt_label = torch.from_numpy(np.array(txt_label)).int()
        except Exception as e:
            return  self.__getitem__(index + 1)
        return img_tensor, txt_len, txt_label, txt_name

    def random_crop(self, img):
        height, width, _ = np.shape(img)
        padding = 10
        crop_img = np.zeros(shape=[height + 2*padding, width + 2*padding, 3], dtype=np.uint8)
        crop_img[padding:padding+height, padding:padding+width, :] = img.copy()
        x = np.random.randint(0,2*padding-1)
        y = np.random.randint(0,2*padding-1)
        img = crop_img[y:y+height, x:x+width, :]
        return img

    def addNoise(self,subImage):
        subImage=self.addSaltNoise(subImage)
        subImage=self.addBlurNoise(subImage)
        return subImage

    def addSaltNoise(self,subImage):
        if np.random.randint(0,10)<8:
            return subImage
        height,width,_=np.shape(subImage)
        rate=1.0*np.random.randint(0,10)/100.0
        for jj in range(int(width*height*rate)):
            row=np.random.randint(0,width-1)
            col=np.random.randint(0,height-1)
            value=np.random.randint(0,255)
            subImage[col][row][:]=value
        return subImage

    def addBlurNoise(self,subImage):
        if np.random.randint(0,10)<8:
            return subImage
        rand=np.random.randint(1,5)
        subImage=cv2.blur(subImage,(rand,rand))
        return subImage



if __name__=='__main__':
    dataset=MyDataSet('./data_sample/',100,32,20)

    trainloader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for batch_id,(img_tensor, txt_len, txt_label, txt_name) in enumerate(trainloader):

        break
