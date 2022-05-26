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
Step1: put the image file into data/, put vgg16_reducedfc.pth into weights/ folder;

Step2: running the train.py;

Step3: eval.py to get mAP
