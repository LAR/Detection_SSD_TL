'''
Created on 10 Mar 2022

@author: daidan
'''

# -*- coding: utf-8 -*-
from __future__ import print_function

import os, sys, zipfile
import urllib.request
import numpy as np
# import skimage.io as io
import pylab
import json
from pycocotools.coco import COCO

ROOT_DIR = os.getcwd()
# print(ROOT_DIR)
Root=ROOT_DIR.split('CornerNet')[0]
print(Root)

def splitJson():

    json_file = Root+'data/coco/annotations/instances.json' # # json源文件
    
    json_file_split = Root+'data/coco/annotations/instances_train.json'
    
    coco=COCO(json_file)
    data=json.load(open(json_file,'r')) 
    
    data_2={}   #新json文件
    
    data_2['type']=data['type']
    data_2['categories']=data['categories']
    
    annotation=[]
    images = []
    # print(data_img['images'])
    
    # imagename = [f for f in os.listdir(os.path.join(dataset_path, image_path))] #读取文件夹下图片名字
    # print(len(data['images'])) 
    imagename=[]
    trainPathTxt=Root+'data/VOC2007/ImageSets/Main/trainval.txt'
    for line in open(trainPathTxt):
        imagename.append(line.strip())
    
    #根据图片数量找到每张图片对应的annotation，即每个‘images’可能有多个annotation（一张图片有多个可识别的目标）
    for name_index in range(0,len(imagename)):
        # 通过imgID 找到其所有instance
        imgID = 0
        for i in range(0,len(data['images'])):
            imageInfo=data['images'][i]
            if imageInfo['file_name'].split(".")[0] == imagename[name_index]:  #根据图片名找到对应的json中的'images'
                imgID=imageInfo['id']
                images.append(imageInfo)
                print(name_index, imgID)  
    
        for ann in data['annotations']:    #根据image_id找到对应的annotation
            if ann['image_id']==imgID:
                annotation.append(ann)
    
    data_2['annotations']=annotation
    data_2['images'] = images
    print(len(data_2['images']))
    # 保存到新的json
    
    json.dump(data_2,open(json_file_split,'w'),indent=4)
    # 从coco标注json中提取单张图片的标注信息

if __name__ == '__main__':
    splitJson()
    