"""
@Filename readfile.py
@Data 2019/08/16
@Author renqiang
@Description read different kinds of file
@Env python3,python2
"""
import struct
import numpy as np
import yaml
import xml.etree.ElementTree as ET

filetype = "yaml"

## read .bin file ##

if filetype == 'bin' :
    data = np.fromfile("test.bin",dtype=np.float32)

    fp = open("test.bin",'rb')
    data = fp.read(4)
    data_float = struct.unpack('f',data)[0]

    fp.close()
    print(data)

## read .yaml file ##
if filetype == 'yaml':
    fp = open('t1.yaml')
    data = yaml.load(fp)
    fp.close()
    fp = open('test.yaml','w')
    data = {1:[['sd',1,4,3,4],['sf',3,4,5,6]],2:[['23',4,5,1,2],['235',6,7,8,9]]}
    yaml.dump(data,fp)
    fp.close()
    print(data)

## read .xml file ##
if filetype == 'xml':
    tree = ET.parse("test.xml")
    if tree.find('size') is not None:
        width = int(tree.find('size').find('width').text)
        height = int(tree.find('size').find('width').text)
        depth = int(tree.find('size').find('depth').text)
        scale_x = 300.0/width
        scale_y = 300.0/height
    
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [max(min(round(scale_x*int(bbox.find('xmin').text)),300),1),
                              max(min(round(scale_y*int(bbox.find('ymin').text)),300),1),
                              max(min(round(scale_x*int(bbox.find('xmax').text)),300),1),
                              max(min(round(scale_y*int(bbox.find('ymax').text)),300),1)]
        objects.append(obj_struct)
    print(objects)