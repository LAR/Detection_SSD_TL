'''
Created on 9 Mar 2022

@author: daidan
'''

# This script is used to evaluate map using gt json and output json in coco format

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

ROOT_DIR = os.getcwd()
# print(ROOT_DIR)
Root=ROOT_DIR.split('CornerNet')[0]
print(Root)

if __name__ == '__main__':
    # coco格式的json文件，原始标注数据
    anno_file = Root+ "data/coco/annotations/instances_test.json"
    res_path = './results/CornerNet/8000/testing/results.json'
    
    coco_gt = COCO(anno_file)
    coco_dt = coco_gt.loadRes(res_path)
    
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
     
    print(cocoEval.stats)
    
    # print(cocoEval.stats[0], cocoEval.stats[12:])