from skimage import img_as_ubyte
from tqdm import tqdm
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import glob
import shutil

def merge(im,c1=0,c2=2):
    res=cv2.bitwise_or(im[:,:,c1],im[:,:,c2])
    return res
    
'''path / image array / folder/ dataframe'''
def show(img,w=6,nc=4,title=None,save=None,cols=['impath'],pltsave=False):
    if type(img) == pd.core.frame.DataFrame:
        if len(img)>20:
            img = img.sample(4,random_state=7)
        
        if title:
            if isinstance(title,str):
                title = img[title].values.tolist()
                print(title)
        img = img[cols].values.tolist()
        img=sum(img,[])
    if type(img)==str:
        if os.path.isdir(img):
            files = os.listdir(img)
            ims = random.sample(files,4)
            img = [os.path.join(img,x) for x in ims]
        else:
            img = cv2.imread(img)
                
    if type(img)==list:
        nc = min(nc,len(img))
        ims=[]
        nr= int(np.ceil(len(img)/nc))
        fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(w*nc, w*nr))
        fig.tight_layout()
        ax = axs.ravel()
        for j,im in enumerate(img):
            if type(im)==str:
                im = cv2.imread(im)    
            else:
                if len(im.shape)==2:
                    im = np.dstack([im]*3)     
            ims.append(im)
            _ = ax[j].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            _ = ax[j].axis('off')  
            if title:
                if len(title)<len(img):
                    r = len(cols)
                    _ = ax[j].set_title(title[j//r],fontsize=10)
                else:
                    _ = ax[j].set_title(title[j])
                
        if save:
            # plt.savefig(save)
            h,w=ims[0].shape[:2]
            ims=[ims[0]]+[cv2.resize(im,(w,h)) for im in ims[1:]]
            img = np.hstack(ims) 
        if pltsave:
            plt.savefig(pltsave,dpi=350,bbox_inches='tight')
        else:
            plt.show()
    else:
        if len(img.shape)==2:
            img = cv2.merge([img]*3)
        plt.figure(figsize=(w,w))
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if title:
            _ = plt.title(title)
        if pltsave:
            plt.savefig(pltsave,dpi=350,bbox_inches='tight')
        else:
            plt.show()
    if save:
        cv2.imwrite(save,img)

def pltshow(ls,w=6,nc=4):
    nc = min(nc,len(ls))
    nr = int(np.ceil(len(ls)/nc))
    H,W = ls[0].shape[:2]
    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(w*nc*W//H, w*nr))
    ax = axs.ravel()
    for j,im in enumerate(ls):
        if len(im.shape)==2:
            _ = ax[j].imshow(im,cmap='gray')
            ax[j].axis('off')
        else:
            _ = ax[j].imshow(im)
            ax[j].axis('off')


def splitdf(df,by=None,test_size=.1,testset=0,total=0):
    if by:
        from sklearn.model_selection import GroupShuffleSplit
        train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7).split(df, groups=df[by]))
        train = df.iloc[train_inds].reset_index(drop=True)
        test = df.iloc[test_inds].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        train,test = train_test_split(df)
    print(train.shape,test.shape)
    if testset:
        train['split']='Validation'
        test['split']='Test'
    else:
        train['split']='Train'
        test['split']='Validation'
    if total:
        return train.append(test).reset_index(drop=True)
    else:
        return train,test
# DEV,TEST=split(df,'eid_ckd')

def samplels(ls,folder):
    print(len(ls))
    import shutil
    os.makedirs(folder,exist_ok=True)
    for im in ls:
        shutil.copy(im,os.path.join(folder,os.path.basename(im)))
import re
def mysearch(p,ftype='(png|jpg|jpeg|tif|bmp)$',contains='',ls=False,imid=False,folder=True,col='impath'):
    ims = []
    for root, directory, files in os.walk(os.path.abspath(p)):
        for file in files:
            pp = os.path.join(root, file)

            if re.search('('+ftype+')',pp, re.IGNORECASE):
                if re.search(contains,pp):
                    ims.append(pp)  
    print('imid:',len(ims))
    if ls:
        if imid:
            ims=[im.split('/')[-1].split('.')[0] for im in ims]
        return ims
    else:
        df = pd.DataFrame()
        df[col] = ims
        # df['filename'] = df['impath'].apply(lambda x:x.split('/')[-1])
        df['imid']=df[col].apply(lambda x:x.split('/')[-1].split('.')[0])
        if folder:
            df['folder']=df[col].apply(lambda x:x.split('/')[-2])
        # df['imid']=df['filename'].apply(lambda x:x.replace('.png',''))
        return df

def mkfile(p):
    os.makedirs(p,exist_ok=True)

def splitim(im,plot=1,res=False):
    b,g,r=cv2.split(im)
    if plot:
        show([b,g,r])
    if res:
        return b,g,r

