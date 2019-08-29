#!/usr/bin/python
# -*- coding: UTF-8 -*-

#coding=utf-8
import numpy as np
import cv2
import caffe
caffe.set_mode_cpu()

model_def = '/home/huangrq/task/person_crowd/person_crowd.prototxt'
model_weights = '/home/huangrq/task/person_crowd/person_crowd.caffemodel' 
net = caffe.Net(model_def,model_weights,caffe.TEST) 
net.save('/home/huangrq/task/person_crowd/person_crowd_new.caffemodel' )

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  
    
transformer.set_raw_scale('data', 255)     
transformer.set_channel_swap('data', (0,1,2)) 

transformer.set_mean('data', np.array([123.765,116.28,103.53]))       
# transformer.set_mean('data', np.array([103.53,116.28,123.765]))   

transformer.set_input_scale('data',np.array([0.01712,0.01751,0.01743])[:,np.newaxis,np.newaxis])
# transformer.set_input_scale('data',np.array([0.01743,0.01751,0.01712])[:,np.newaxis,np.newaxis])

image=caffe.io.load_image('/home/huangrq/task/CSRNet-pytorch-crowd-ssmi/IMG_1.jpg')
transformed_image=transformer.preprocess('data',image)

net.blobs['data'].data[...] = transformed_image
out_put = net.forward()
print(out_put['density_map'].sum())