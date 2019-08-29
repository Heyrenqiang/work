# -*- coding=utf-8 -*-
'''
@Filename: caffe_inference.py
@Author: renqiang
@Data: 2019-08-12
@Description: run images on whole mssd model, save the results 
@Env requirements: python2;caffe installed;
'''

import caffe
import time
import numpy as np
import cv2
import glob
import os
import yaml


class NetOperator(object):
	def __init__(self,model_def,model_weights,gpu_id,channel_swap,mean_value,scale):
		'''
		Set network param and init network
		'''
		self.input_size = []
		self.mean_value = mean_value
		self.mean_image = []
		self.scale = scale
		self.channel_swap = channel_swap

		self.set_channel()

		if gpu_id < 0:
			caffe.set_mode_cpu()
		else:
			caffe.set_device(gpu_id)
			caffe.ser_mode_gpu()
	
		if self.mean_value != '':
			self.set_mean()
		
		if self.scale != '':
			self.set_scale()
		

	
		self.net = caffe.Net(model_def,model_weights,True)
	
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data',(2,0,1))	
		self.transformer.set_raw_scale('data',255)
		self.transformer.set_channel_swap('data',self.channel_swap)
		self.transformer.set_mean('data',self.mean_value)
		self.transformer.set_input_scale('data',self.scale)	

	
	def forward(self,image):

		blobs = self.preprocess(image)
		self.net.blobs['data'].data[...] = blobs
		output_blobs = self.net.forward()
		return output_blobs
	 
	def preprocess(self,image):

		transformed_image = self.transformer.preprocess('data',image)
		return transformed_image

	def set_channel(self):
		channel_swap = self.channel_swap.strip().split(',')
		self.channel_swap = tuple(map(int,channel_swap))

	def set_mean(self):
		
		mean_value = self.mean_value.strip().split(',')
		mean_value = list(map(float,mean_value)) 
		self.mean_value = np.array(mean_value)

	def set_scale(self):
		
		scale = self.scale.strip().split(',')
		scale = list(map(float,scale))
		scale = np.array(scale)
		scale = 1.0/scale
		self.scale = scale[:,np.newaxis,np.newaxis]

start = time.time()


model_def = '/home/huangrq/task/scripts/caffe_inference/models/mssd_with_detectionout.prototxt'
model_weights = '/home/huangrq/task/scripts/caffe_inference/models/mssd_with_detectionout.caffemodel'
gpu_id = -1
channel_swap = '2,1,0'
mean_value = '127,127,127'
scale = '255,255,255'
image_root = '/home/huangrq/DataSet/testdata/image_300x300'
fp = open('mssd_with_det_result.yaml','w')

results = {}
num = 0
for image_path in glob.glob(image_root+'/*'):
	image_name = os.path.basename(image_path)
	results[image_name] = []
	image = caffe.io.load_image(image_path)
	net = NetOperator(model_def,model_weights,gpu_id,channel_swap,mean_value,scale)
	outputs = net.forward(image)
	out = outputs['detection_out']
	out = np.squeeze(outputs['detection_out'],axis = (0,1))
	for i in range(out.shape[0]):   
		result = {}
		result['class_id'] = int(out[i,1])
		result['score'] = round(out[i,2],3)
		result['x_min'] = round(300*out[i,3],1)
		result['y_min'] = round(300*out[i,4],1)
		result['x_max'] = round(300*out[i,5],1)
		result['y_max'] = round(300*out[i,6],1)
		results[image_name].append(result)

	num = num+1
	# if num == 100 :
	# 	yaml.dump(results,fp)
	# 	break

	print('finish  {}............'.format(num))

    # img = cv2.imread(image_path)
    # for i in range(out.shape[0]):
    #     p_left_top = (int(out[i,3]*300),int(out[i,4]*300))
    #     p_right_bottom = (int(out[i,5]*300),int(out[i,6]*300))
    #     cv2.rectangle(img,p_left_top,p_right_bottom,(255,255,0),2)
    # cv2.imshow('',img)
    # k = cv2.waitKey(1)

yaml.dump(results,fp)
	
fp.close()
end = time.time()
time_consumption = end-start
print("time consumption: ",time_consumption)
