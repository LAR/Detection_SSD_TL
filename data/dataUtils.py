'''
Created on Nov 18, 2020

@author: daidan
'''
import random
import os
from typing import List, Any
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
from numpy import number
from collections import Counter

def dataName():
    trainval_percent = 0.7  
    train_percent = 0.8
    xmlfilepath = './Fruit/Annotations/'
    txtsavepath = './Strawberry/ImageSets/Main'  
    total_xml = os.listdir(xmlfilepath)  
    
    num = len(total_xml)  
    list = range(num)  
    tv = int(num*trainval_percent)  
    tr = int(tv*train_percent)  
    trainval = random.sample(list,tv)  
    train = random.sample(trainval,tr)  
    
    ftrainval = open(txtsavepath+'/trainval.txt', 'w')  
    ftest = open(txtsavepath+'/test.txt', 'w')  
    ftrain = open(txtsavepath+'/train.txt', 'w')  
    fval = open(txtsavepath+'/val.txt', 'w')  
    
    for i in list:  
        name = total_xml[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)
#             json.dump(name,ftrainval)  
            if i in train:  
                ftrain.write(name)  
#                 json.dump(name,ftrain)
            else:  
                fval.write(name) 
#                 json.dump(name,fval) 
        else:  
            ftest.write(name) 
#             json.dump(name,ftest) 
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest .close()
    print('save name')
 
def jsonSplit():
    
    json_file = json.load(open('all.json', "r", encoding="utf-8"))
    jsonInfo=[]
    for eachJson in json_file:
        jsonInfo.append(eachJson)
        imageName=eachJson['External ID']
        print(imageName)
        filename='./json/'+imageName.split('.')[0]+'.json'
        with open(filename,'w',encoding = 'utf-8') as f:
                #print(len(data_list))
                data = json.dumps(jsonInfo)
                f.write(data)
                jsonInfo = []
    
    
def deleteNoLableImage():
    #read the label file
    label_path = "/Users/daidan/Documents/data/tomato_497/label"
    #get the json file name 
    for root, dirs, files in os.walk(label_path): 
        print(files)
        for name in files:
            print(name)
            dataFile="/Users/daidan/Documents/data/tomato data/"+name.split('.')[0]+'.jpg'
            shutil.copy(dataFile,"/Users/daidan/Documents/data/tomato_497/JPEGImages/")
        

def jsonXmlLable():
    # 1.标签路径
    labelme_path = "./VOC2007/Annotations/json/"
    #原始labelme标注数据路径
    saved_path = "./VOC2007/"
    # 保存路径
    isUseTest=True#是否创建test集
    # 2.创建要求文件夹
#     if not os.path.exists(saved_path + "Annotations"):
#         os.makedirs(saved_path + "Annotations")
#     if not os.path.exists(saved_path + "JPEGImages/"):
#         os.makedirs(saved_path + "JPEGImages/")
#     if not os.path.exists(saved_path + "ImageSets/Main/"):
#         os.makedirs(saved_path + "ImageSets/Main/")
    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
    files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]
    print(files)
    # 4.读取标注信息并写入 xml
    for json_file_ in files:
        json_filename = labelme_path + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread('./VOC2007/JPEGImages/' + json_file_ + ".jpg").shape
        with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
     
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'tomato_data' + '</folder>\n')
            xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>tomato Data</database>\n')
            xml.write('\t\t<annotation>tomato</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>tomato</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for multi in json_file["shapes"]:
                if multi["shape_type"]=='rectangle':
                
                    points = np.array(multi["points"])
                    labelName=multi["label"]
                    xmin = min(points[:, 0])
                    xmax = max(points[:, 0])
                    ymin = min(points[:, 1])
                    ymax = max(points[:, 1])
                    label = multi["label"]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>' + labelName+ '</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                        print(json_filename, xmin, ymin, xmax, ymax, label)
            xml.write('</annotation>')
    # 5.复制图片到 VOC2007/JPEGImages/下
#     image_files = glob("labelmedataset/images/" + "*.jpg")
#     print("copy image files to VOC007/JPEGImages/")
#     for image in image_files:
#         shutil.copy(image, saved_path + "JPEGImages/")
    # 6.split files for txt
    txtsavepath = saved_path + "ImageSets/Main/"
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    total_files = glob("./VOC2007/Annotations/*.xml")
    total_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in total_files]
    trainval_files=[]
    test_files=[] 
    if isUseTest:
        trainval_files, test_files = train_test_split(total_files, test_size=0.15, random_state=55) 
    else: 
        trainval_files=total_files 
    for file in trainval_files: 
        ftrainval.write(file + "\n") 
    # split 
    train_files, val_files = train_test_split(trainval_files, test_size=0.15, random_state=55) 
    # train
    for file in train_files: 
        ftrain.write(file + "\n") 
    # val 
    for file in val_files: 
        fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    
def LableBoxToXml():
    # 1.标签路径
    labelbox_path = "/Users/daidan/Documents/data/tomato_497/label/"
    #原始labelme标注数据路径
    saved_path = "/Users/daidan/Documents/data/tomato_497/"
    # 保存路径
    isUseTest=True#是否创建test集
    # 2.创建要求文件夹
    if not os.path.exists(saved_path + "Annotations"):
        os.makedirs(saved_path + "Annotations")
    if not os.path.exists(saved_path + "JPEGImages/"):
        os.makedirs(saved_path + "JPEGImages/")
    if not os.path.exists(saved_path + "ImageSets/Main/"):
        os.makedirs(saved_path + "ImageSets/Main/")
    # 3.获取待处理文件
    files = glob(labelbox_path + "*.json")
    files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]
#     print(files)
    # 4.读取标注信息并写入 xml
    for json_file_ in files:
        json_filename = labelbox_path + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread('/Users/daidan/Documents/data/tomato_497/JPEGImages/' + json_file_ + ".jpg").shape
        xmlFile=saved_path + "Annotations/" + json_file_ + ".xml"
        with codecs.open(xmlFile, "w", "utf-8") as xml:
     
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'Tomata_497_data' + '</folder>\n')
            xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>Tomata Data</database>\n')
            xml.write('\t\t<annotation>rectangle</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>' + str(json_file[0]['ID']) + '</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>tomato</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for multi in json_file[0]['Label']['objects']:
                if multi["title"]=='Tomato - D':
                
                    points = np.array(multi["bbox"])
                    labelName=multi["value"]
                    points=points.reshape(1,1)[0][0]
                    xmin = points['left']
                    xmax = points['left']+points['width']
                    ymin = points['top']
                    ymax = points['top']+points['height']
                    label = multi["featureId"]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>Tomato_D</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')
                        print(json_filename, xmin, ymin, xmax, ymax, label)
            xml.write('</annotation>')
    # 5.复制图片到 VOC2007/JPEGImages/下
#     image_files = glob("labelmedataset/images/" + "*.jpg")
#     print("copy image files to VOC007/JPEGImages/")
#     for image in image_files:
#         shutil.copy(image, saved_path + "JPEGImages/")
    # 6.split files for txt
    txtsavepath = saved_path + "ImageSets/Main/"
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    total_files = glob(saved_path + "Annotations/" + "*.xml")
    total_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in total_files]
    trainval_files=[]
    test_files=[] 
    if isUseTest:
        trainval_files, test_files = train_test_split(total_files, test_size=0.2, random_state=55) 
    else: 
        trainval_files=total_files 
    for file in trainval_files: 
        ftrainval.write(file + "\n") 
    # split 
    train_files, val_files = train_test_split(trainval_files, test_size=0.2, random_state=55) 
    # train
    for file in train_files: 
        ftrain.write(file + "\n") 
    # val 
    for file in val_files: 
        fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

def preLocation():
    
    saved_path = "./VOC2007/"
    txtsavepath = saved_path + "ImageSets/preMain/"
    isUseTest=True #是否创建test集
    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest = open(txtsavepath + '/test.txt', 'w')
    ftrain = open(txtsavepath + '/train.txt', 'w')
    fval = open(txtsavepath + '/val.txt', 'w')
    Image_files = glob(saved_path + "Annotations/" + "*.xml")
    Image_files = [i.replace("\\","/").split("/")[-1].split(".xml")[0] for i in Image_files]
    

    WF_files = glob(saved_path + "WFruit/" + "*.jpg")
    # WFruit29 WFruit81 383 
    WF_files=[i.split(".")[1].split("/")[-1] for i in WF_files]
    total_files=WF_files+Image_files
    
    trainval_files=[]
    test_files=[] 
    if isUseTest:
        trainval_files, test_files = train_test_split(total_files, test_size=0.2, random_state=55) 
    else: 
        trainval_files=total_files 
    for file in trainval_files: 
        ftrainval.write(file + "\n") 
    # split 
    train_files, val_files = train_test_split(trainval_files, test_size=0.2, random_state=55) 
    # train
    for file in train_files: 
        ftrain.write(file + "\n") 
    # val 
    for file in val_files: 
        fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def rename():
    
    img_path = '/Users/daidan/Documents/data/stem/'
    imglist = glob(os.path.join(img_path, '*.jpeg'))
    print(imglist)
    i = 488
    for img in imglist:
        i+=1
        src = os.path.join(os.path.abspath(img_path), img) #原先的图片名字
        dst = os.path.join(os.path.abspath(img_path), 'WFruit' +str(i)+'.jpg') #根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        os.rename(src, dst) #重命名,覆盖原先的名字


def imgShapeNum():
    file1='./VOC2007/JPEGImages/'
    file2='./VOC2007/WFruit/'
    
    imglist = glob(file1 + "*.jpg")
    numberList=[]
    imgSizeList=[]
#     for imgFile in imglist:
#         im = cv2.imread(imgFile)
#         imgSize=im.shape
#         
#         if imgSize[1]==3888:
#             numberList.append(0)
#         
#         if imgSize[1]==5312:
#             numberList.append(1)
# 
#     numberList=Counter(numberList) #Counter({0: 467, 1: 29})
#    print(numberList)

    for imgFile in glob(file2 + "*.jpg"):
        im = cv2.imread(imgFile)
        imgSize=im.shape
        imgSizeList.append(imgSize)
        
    print(imgSizeList)

if __name__ == '__main__':
#     rename()
#     preLocation()
    jsonSplit()




    