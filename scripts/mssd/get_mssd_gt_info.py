"""
@Filename readfile.py
@Data 2019/08/27
@Author renqiang
@Description caculate the MAP for detection task
@Env python3,python2
"""

import numpy as np
import yaml
import xml.etree.ElementTree as ET
import glob
import os
import cv2

def get_gt(img_name):
    tree = ET.parse(img_name)
    if tree.find('size') is not None:
        width = int(tree.find('size').find('width').text)
        height = int(tree.find('size').find('height').text)
        depth = int(tree.find('size').find('depth').text)
        scale_x = 300.0/width
        scale_y = 300.0/height
    
    objects = {'background':[],'car':[],'tricycle':[],'pedestrian':[],'rider':[]}
    box_num = {'background':0,'car':0,'tricycle':0,'pedestrian':0,'rider':0}
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        if obj_struct['difficult'] == 0:
            box_num[obj_struct['name']] = box_num[obj_struct['name']]+1
        obj_struct['det'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [max(min(round(scale_x*int(bbox.find('xmin').text)),300),1),
                              max(min(round(scale_y*int(bbox.find('ymin').text)),300),1),
                              max(min(round(scale_x*int(bbox.find('xmax').text)),300),1),
                              max(min(round(scale_y*int(bbox.find('ymax').text)),300),1)]
        objects[obj_struct['name']].append(obj_struct)
    return objects,box_num


if __name__ == '__main__':
    classnum = 5
    gt_label_root = '/home/huangrq/DataSet/testdata/annotations/'
    image_root = '/home/huangrq/DataSet/testdata/image_300x300/'
    ovthresh = 0.5
    use_difficult = 0
    labels = ['background','car','tricycle','pedestrian','rider']

    class_det = {i:[] for i in labels}
    gt_boxs_info = {}
    gt_box_num_class = {i:0 for i in labels}
    num = 0

    for image_name in glob.glob(image_root+'*'):
        image_name = os.path.basename(image_name)
        gt_path = gt_label_root+image_name.replace('.jpg','.xml')
        gt_boxs_info[image_name],one_gt_box_num = get_gt(gt_path)
        for key,value in one_gt_box_num.items():
            gt_box_num_class[key] = gt_box_num_class[key]+value
        num = num+1
        # if num == 100 :
        #     break
    
    fp = open('gt_boxs_info.yaml','w')
    yaml.dump(gt_boxs_info,fp)
    fp.close()

    # for image_name,gt_box_info in gt_boxs_info.items():
    #     image = cv2.imread(image_root+image_name)
    #     for label in labels:
    #         for class_box in gt_box_info[label]:
    #             left_top = (int(class_box['bbox'][0]),int(class_box['bbox'][1]))
    #             right_bottom = (int(class_box['bbox'][2]),int(class_box['bbox'][3]))
    #             cv2.rectangle(image,left_top,right_bottom,(255,255,0),2)
    #             cv2.putText(image,label,(left_top[0]+2,left_top[1]+6),cv2.FONT_HERSHEY_COMPLEX,0.2,(255,255,0),1,cv2.LINE_AA)
    #     cv2.imshow('',image)
    #     k = cv2.waitKey(3000)

    print(gt_box_num_class)
    