import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
import pandas as pd
import cv2
from skimage import img_as_ubyte
import torch
import torchvision.transforms as transforms
import random
import numpy as np
from skimage.morphology import skeletonize, thin
from scipy import ndimage
import copy
from options.test_options import TestOptions
import sys
from torchvision.transforms._transforms_video import NormalizeVideo
from fundus_util import *

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.w=opt.w
        self.B=opt.B
        self.A=opt.A
        df = opt.df.reset_index(drop=True)
        self.df =df
        self.idxs=df.pid_eye.unique()
        print('*************************',df.shape,df.pid_eye.nunique(),'*************************')
    def __getitem__(self, index):    
        
        s=self.df[self.df.pid_eye==self.idxs[index]]
        s = s.groupby('Phase').sample(4,replace=True)#For each modality, sample 4 frames
        s=s.drop_duplicates()
        max_len=len(s)
        s = s.sort_values('level_1')[:max_len]#Sort in time
        
        params = get_params(self.opt, (self.w,self.w))
        transform_A = get_transform(self.opt, params)

        paths = s['fa_rigid'].values.tolist()
        addco=1
        if addco:
            A_path = s.co_rigid.values[0]
            paths=[A_path]+paths
            mask = img_as_ubyte(cv2.imread(paths[0])[:,:,1]>11)
        else:
            mask =  img_as_ubyte(cv2.imread(paths[0])[:,:,1]>2)
        for i, p in enumerate(s.drop_duplicates()['fa_rigid'].values.tolist()):
            fa = cv2.imread(p)[:,:,1]
            mask_=img_as_ubyte(fa>2)
            mask=cv2.bitwise_and(mask,mask_)
        
        mask=remove_small_objects(mask,keeplargest=1)
        mask=dilate(mask,11)
        mask=ndimage.binary_fill_holes(mask).astype(np.uint8)

        tensors=[]
        if not self.opt.no_flip:
            aug = 0
        if len(paths)<max_len+1:
            paths=paths+[paths[-1]]*(max_len+1-len(paths))
        for i, p in enumerate(paths):
            fa = cv2.imread(p)
            if (i==0) and (addco==1):
                fa= cv2.cvtColor(fa,cv2.COLOR_BGR2RGB)
            else:
                fa=fa[:,:,1]
                fa=cv2.merge([fa]*3)
            
            fa[mask==0]=0

            B=crop_image_from_mask(fa,mask)
            if not self.opt.no_flip:
                if aug>0:
                    try:
                        if i==0:
                            h,w=B.shape[:2]
                            x = random.randint(0,w//(1.5))
                            x2=  random.randint(x+w//3,w)
                            y = random.randint(0, h//(1.5))
                            y2 = random.randint(y+h//3, h)
                        B = B[y:y2,x:x2,:]
                    except:
                        aug=0

            B=cv2.resize(B,(self.w,self.w))

            if i==1:
                image1 = B
            if i==len(paths)-1:
                image2 = B

            B=Image.fromarray(B)
            if i==0:
                tensor = transform_A(B)
            else:
                tensor = transforms.ToTensor()(B)
            tensors.append(tensor)
        images = torch.stack(tensors[1:])
        diff = cv2.absdiff(image1[:,:,1],image2[:,:,1])
        _, map = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        map = transforms.ToTensor()(map)

        images = NormalizeVideo(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(images.permute(1, 0, 2, 3)).permute(1,0,2,3)
        images = torch.cat([tensors[0].unsqueeze(0), images], dim=0)
        inst_tensor = feat_tensor = 0
        input_dict = {'label': tensors[0], 'inst': inst_tensor, 'image': images, 
                      'feat': feat_tensor, 'path': self.idxs[index], 'tempo_map': map}    

        return input_dict

    def __len__(self):
        return len(self.idxs)

    def name(self):
        return 'AlignedDataset'
