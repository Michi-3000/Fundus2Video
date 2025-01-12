import os
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
# from skimage.morphology import skeletonize

def get_center(mask):
    dcs, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dcs = sorted(dcs, key=cv2.contourArea)
    cnt=dcs[-1]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy
################################################
def readvideo(path):
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS)

    size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    C=np.empty((size[1], size[0], fNUMS))
    i=0
    while(cap.isOpened()):
        ret, frame=cap.read()
        if ret==True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            C[:, :, i] = frame_gray
            i=i+1
        else:
            break
    return C, fps, size
def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def gray2rgb(img,c=None):
    if c:
        h,w=img.shape[:2]
        mask=np.zeros((w,h,3),dtype=np.uint8)
        mask[:,:,c]=img
        return mask
    else:
        return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def close(im,k=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(s,s))
    im=cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return im

def erode(im,k=5):
    # kernel = np.ones((k,k),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    im = cv2.erode(im,kernel,iterations = 1)
    return im
def dilate(im,k=5):
    # kernel = np.ones((k,k),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    im = cv2.dilate(im,kernel,iterations = 1)
    return im

import math
# rescale the images to have the same radius (300 pixels or 500 pixels)
# default scale = 300
def scaleRadius(img, scale):
    x = img[math.ceil(img.shape[0] / 2), :, :].sum(1)
    r = ( x > x.mean() / 10 ).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)
def subtract_average_color(image, scale=300):
    try:
        # scale img to a given radius
        scale_img = scaleRadius(image, scale) # scale_img
        # remove outer 10%
        null_img = np.zeros(scale_img.shape) # create null rgb imgage
        cv2.circle(null_img, (scale_img.shape[1] // 2, scale_img.shape[0] // 2), int(scale *0.9), (1, 1, 1), -1, 8, 0)
        # subtract local mean color
        processed_img = cv2.addWeighted(scale_img, 4, cv2.GaussianBlur(scale_img, (0, 0), scale / 30), -4, 128) * null_img + 128 * (1 - null_img)
        return processed_img
    except BaseException as e:
        print(e)
        return None
        
def subtract_average_colorl(img,r=.95,plot=1):        
    h,w=img.shape[:2]
    scale=w
    scale_img=img.copy()
        # mask = img_as_ubyte(img[:,:,1]>7)
        # null_img=cv2.merge([mask]*3)
    null_img = np.zeros(scale_img.shape) # create null rgb imgage
    null_img=cv2.circle(null_img, (scale_img.shape[1] // 2, scale_img.shape[0] // 2), int(scale * r/2), (1, 1, 1), -1)#, 8, 0)
        # subtract local mean color
    processed_img = cv2.addWeighted(scale_img, 4, cv2.GaussianBlur(scale_img, (0, 0), scale / 30), -4, 128) * null_img + 128 * (1 - null_img)
    processed_img=processed_img.astype(np.uint8)
    if plot:
        from show import show
        show([img,processed_img],w=10)
    return processed_img

# FOV = np.zeros((w,w),dtype=np.uint8)
# FOV = cv2.circle(FOV,(w//2,w//2),w//2-1,(255,255,255),-1)
# method1
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
def crop_image_from_mask(img,mask):
    # img[mask==0]=0
    if img.ndim ==2:
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img   
      
def croppairs(im1,im2):
    maska=img_as_ubyte(im1[:,:,1]>7)
    maskb=img_as_ubyte(im2[:,:,1]>7)
    mask=cv2.bitwise_and(maska,maskb)
    mask=255-remove_small_objects(255-mask,keeplargest=True)
    # im1[mask==0]=0
    # im2[mask==0]=0
    im1=crop_image_from_mask(im1,mask)
    im2=crop_image_from_mask(im2,mask)
    return im1,im2,mask
def crop_image_with_stat(im,th=15,keeplargest=False):
    mask = img_as_ubyte(im[:,:,2]>th)
    mask = remove_small_objects(mask,100,keeplargest=keeplargest)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(cnt) for cnt in contours]

    top_x = min([x for (x, y, w, h) in rects])
    top_y = min([y for (x, y, w, h) in rects])
    bottom_x = max([x+w for (x, y, w, h) in rects])
    bottom_y = max([y+h for (x, y, w, h) in rects])

    crop = im[top_y:bottom_y, top_x:bottom_x]
    # show([mask,crop],w=4)
    return crop,[top_y,bottom_y,top_x,bottom_x]

def overlay(im1,im2,w1=1,w2=0.5,plot=1):
    
    if im2.shape!=im1.shape:
        if len(im2.shape)>len(im1.shape):
            im1=cv2.merge([im1]*3)
        elif len(im2.shape)<len(im1.shape):
            im2=cv2.merge([im2]*3)
        if im2.shape!=im1.shape:
            h,w=im1.shape[:2]
            im2=cv2.resize(im2,(w,h))
    mer=cv2.addWeighted(im1,w1,im2,w2,0)    
    if plot:
        from show import show
        show(mer)
    else:
        return mer

def rmbackground(im,mask):
    if mask.dtype.name == 'bool':
        mask=img_as_ubyte(mask)
    im=cv2.bitwise_and(im,im,mask=mask)  
    return im

def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)    
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border  
#############################################################################
# method2
def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-7
    _, mask = cv2.threshold(gray_img, max(0,threhold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask

def _get_center_radius_by_hough(mask):
    circles= cv2.HoughCircles((mask*255).astype(np.uint8),cv2.HOUGH_GRADIENT,1,1000,param1=5,param2=5,minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2+1)
    center = circles[0,0,:2]
    radius = circles[0,0,2]
    return center,radius

def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    return center_mask    

def get_mask(img):
    if img.ndim ==3:
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    center, radius = _get_center_radius_by_hough(tmp_mask)
    #resize back
    center = [center[1]*2,center[0]*2]
    radius = int(radius*2)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius

def mask_image(img,mask):
    img[mask<=0,...]=0
    return img

def FOV(im): 
    mask,bbox,center,radius=get_mask(im)
    r_img=mask_image(im,mask)
    r_img,border=remove_back_area(r_img,bbox=bbox)
    im,sup_border=supplemental_black_area(r_img)
    return im
    
def resize(i):
    p = df.loc[i,'im_path']
    filename =  df.loc[i,'filename']#.replace('jpg','png')
    try:
        org = cv2.imread(p)
        org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
        shape = org.shape
#         if shape == (1080, 1620, 3):
#         r_img = crop_image_from_gray(org)
#         im,sup_border=supplemental_black_area(r_img)
        im = FOV(org)
        im = cv2.resize(im,(512,512))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out+filename,im)
        return (shape[0],shape[1])
    except Exception as e:
        print(e,filename)
        return ''

# method3	
def crop(input_image):
    _, mask = cv2.threshold(cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY), 10, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = list(cv2.contourArea(contour) for contour in contours)
    index = np.where(areas == np.max(areas))[0][0]
    x0, y0, w, h = cv2.boundingRect(contours[index])
    mask = cv2.fillPoly(img=np.zeros(shape=mask.shape,dtype=np.uint8), pts=[contours[index]], color=1)
    res_img = np.empty_like(input_image)
    for j in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            for k in range(input_image.shape[2]): 
                res_img[j, i, k] = input_image[j, i, k] * mask[j, i]
    
    return res_img[y0:y0+h, x0:x0+w, :], mask[y0:y0+h, x0:x0+w], [x0, y0, w, h]


def remove_small_objects(img, min_size=50,keeplargest=False,returnN=False):
        # find all your connected components (white blobs in your image)
    try:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        #taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        maxs=max(sizes)
        N=nb_components
        for i in range(0, nb_components):
            if keeplargest:
                if sizes[i] < maxs:
                    img2[output == i + 1] = 0
                    N-=1	
            elif sizes[i] < min_size:
                img2[output == i + 1] = 0
                N-=1	
        if returnN:
            return img2,N  
        else:
            return img2
    except:
        # print('')
        return img
def rmshort(im,minlen=20):
    dcs, hierarchy= cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    out=im.copy()
    for cnt in dcs:
        # print(cv2.arcLength(cnt,closed=False))
        if cv2.arcLength(cnt,closed=False)<minlen:
            out = cv2.drawContours(out, [cnt], -1,0, -1)
    return out

def extract(filled):
    a = np.zeros(filled.shape,dtype=np.uint8)
    v = np.zeros(filled.shape,dtype=np.uint8)
    a[(filled==1)|(filled==3)]=1
    v[(filled==2)|(filled==3)]=1
    return a,v    

def fillwith(ve,a,v,avp,avconth,minavArc):
    from scipy import ndimage as ndi  
    from skimage.segmentation import watershed
    from skimage import img_as_ubyte
    crossing=cv2.bitwise_and(a,v)
    # crossing[disc>0]=0
    
    skela=img_as_ubyte(skeletonize(avp[:,:,2]>avconth))
    skelv=img_as_ubyte(skeletonize(avp[:,:,0]>avconth))
    a=cv2.bitwise_or(skela,a)
    v=cv2.bitwise_or(skelv,v)
    # ve=cv2.bitwise_or(a,v)
    a=rmshort(a,minavArc)
    v=rmshort(v,minavArc)

    distance = ndi.distance_transform_edt(ve)
    label = np.zeros_like(ve)
    label[a>0]=1
    label[v>0]=2
    label[crossing>0]=3
    filled = watershed(-distance, markers=label,
                    mask=ve!=0,compactness=0)
    a,v = extract(filled)
    bgr=cv2.merge([v*255,avp[:,:,1],a*255])
    return bgr