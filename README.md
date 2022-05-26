# Detection_SSD_TL


## Fruit Dataset Structure

Numeric values of the labelled coordinates are stored in .xml files. All the label coordinates in train dataset are stored in a single .xml file. The zip file in Google Dirve mainly contain:

```bash
├── Annotations
│   ├── *.xml (those files containing image label coordinates for images )
├── ImageSets
│   ├── *.txt (the poinsettia for training and testing)
├── JPEGImages
│   ├── *.JPEG (the poinsettia images)
```
## Running the SSD code
Step1: put the image file into data/, put vgg16_reducedfc.pth into new weights/ folder;

Step2: running the train.py;

Step3: eval.py to get mAP

## About the Transfer Learning
The model path of the source domain is args.resume in trainLabelNum.py

checkpoint = torch.load(args.resume)
ssd_net.load_state_dict(checkpoint)

For the multi-class, like multiFruit dataset, the SSD structure is different with single-class:

if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        preTrain_dict = torch.load(args.resume,map_location=torch.device('cpu'))
        SSD_dict=ssd_net.state_dict()
        
        for i in range(0,6):
            
            weightName='conf.'+str(i)+'.weight'
            bias='conf.'+str(i)+'.bias'        
            # conf_weight=preTrain_dict[weightName]
            # conf_bias=preTrain_dict[bias]
            
            newWeigh=SSD_dict[weightName]
            newBias=SSD_dict[bias]
            
            preTrain_dict[weightName]=newWeigh
            preTrain_dict[bias]=newBias
        
        ssd_net.load_state_dict(preTrain_dict)
        print('SSD load model')

## About the CornerNet

https://github.com/princeton-vl/CornerNet
