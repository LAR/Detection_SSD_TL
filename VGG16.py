'''
Created on Feb 3, 2021

@author: daidan
'''

import json
import os
from torchvision import transforms
import torch
from sklearn.metrics import accuracy_score
from dataset import Pre_Trainset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def VGGPre():
    
    lr = 0.005
    batch_size=16

    pretrained_net = models.vgg16(pretrained=True).to(device)
    
    pretrained_net.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 2), nn.Softmax(dim=1))
    
    pretrained_net.to(device)
    
    
    optimizer = optim.SGD(pretrained_net.parameters(),lr=lr, weight_decay=0.001)
    
    
    imageFile='./data/VOC2007/JPEGImages/'
    WFruit='./data/VOC2007/WFruit/'
    
    preTrain_list=[]
    preTest_list=[]
    
    trainTestFile='./data/VOC2007/ImageSets/preMain/'
    
    trainImage=trainTestFile+'trainval.txt'
    testImage=trainTestFile+'test.txt'
    
    for name in open(trainImage, "r"):
        name=name.strip('\n')
#         print(name)
        if 'WFruit' in name:
            path=WFruit+name+'.jpg'
            preTrain_list.append(path)
        else:
            path=imageFile+name+'.jpg'
            preTrain_list.append(path)
    
    for name in open(testImage, "r"):
        name=name.strip('\n')
        if 'WFruit' in name:
            path=WFruit+name+'.jpg'
            preTest_list.append(path)
        else:
            path=imageFile+name+'.jpg'
            preTest_list.append(path)
    
    
    train_loader = torch.utils.data.DataLoader(Pre_Trainset(preTrain_list,train=True),batch_size)
    
    test_loader = torch.utils.data.DataLoader(Pre_Trainset(preTest_list, train=False),batch_size)
    

        
    criterion = nn.CrossEntropyLoss().to(device)    
    
    lossValues=[]
    trainAccs=[]
    testAcc=[]
    for epoch in range(20):  # loop over the dataset multiple times
        train_acc = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            
#             print(len(train_loader))
            # get the inputs
            inputs, labels = data
            
            inputs=inputs.to(device)
            labels=labels.to(device)
    
            # forward + backward + optimize
            outputs = pretrained_net(inputs)
           
            loss = criterion(outputs, labels.squeeze().long())
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            predict=outputs.detach().cpu().numpy()
           
            train_acc += accuracy_score(predict.argmax(axis=1), labels.detach().cpu().numpy())
            
             
#             print('[%d, %5d] loss: %.3f acc:%.3f' %
#                   (epoch + 1, i + 1, running_loss, accuracy2))
        lossValue=running_loss/len(train_loader)
        lossValues.append(lossValue)
        
        trainAcc=train_acc/len(train_loader)
        trainAccs.append(trainAcc)
        
        print('Epoch %d. Loss: %f, Train acc %f' % (epoch, lossValue, trainAcc))
        
        test_acc = 0
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                
                images=images.to(device)
                labels=labels.to(device)
                
                outputs = pretrained_net(images)
                
                predict=outputs.detach().cpu().numpy()
               
                test_acc += accuracy_score(predict.argmax(axis=1), labels.detach().cpu().numpy())
                
    
            vaule=test_acc / len(test_loader)
            testAcc.append(vaule)
            print('Accuracy of the network on the test images: %d %%' % (
                100 * vaule))
    
    torch.save(pretrained_net.state_dict(),'./weights/preVGG16Model.pth',_use_new_zipfile_serialization=False)
            
    with open('lossPre.txt', 'w') as f:
        f.write(str(lossValues))
    with open('trainPre.txt', 'w') as f:
        f.write(str(trainAccs))
    with open('testPre.txt', 'w') as f:
        f.write(str(testAcc))


if __name__ == '__main__':
    
    VGGPre()