# -*- coding=utf-8 -*-
'''
@Filename: caffe_inference.py
@Author: renqiang
@Data: 2019-08-29
@Description: run images on mssd with detection_out layer manually added, save the results
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


model_def = '/home/huangrq/task/mssd/mssd.prototxt'
model_weights = '/home/huangrq/task/mssd/mssd.caffemodel'
gpu_id = -1
channel_swap = '2,1,0'
mean_value = '127,127,127'
scale = '255,255,255'
image_root = '/home/huangrq/DataSet/testdata/image_300x300'
fp = open('mssd_no_det_result.yaml','w')

results = {}
num = 0
for image_path in glob.glob(image_root+'/*'):
    image_name = os.path.basename(image_path)
    results[image_name] = []
    image = caffe.io.load_image(image_path)
    net = NetOperator(model_def,model_weights,gpu_id,channel_swap,mean_value,scale)
    outputs = net.forward(image)
    np.set_printoptions(precision=4)
    loc = outputs['mbox_loc_concat']
    conf = outputs['mbox_conf_flat']
    pri = outputs['mbox_priorbox_concat']
    def decode_all_boxs(pri,loc):
        decoded_boxs = []
        variance_exist = True
        for i in range(loc.shape[1]/4):

            pri_center_x = (pri[0,0,4*i+0]+pri[0,0,4*i+2])/2.0
            pri_center_y = (pri[0,0,4*i+1]+pri[0,0,4*i+3])/2.0
            pri_width = (pri[0,0,4*i+2]-pri[0,0,4*i+0])
            pri_height = (pri[0,0,4*i+3]-pri[0,0,4*i+1])

            if variance_exist:
                decoded_box_center_x = pri_center_x + pri[0,1,4*i+0]*loc[0,4*i+0]*pri_width
                decoded_box_center_y = pri_center_y + pri[0,1,4*i+1]*loc[0,4*i+1]*pri_height
                decoded_box_width = np.exp(pri[0,1,4*i+2]*loc[0,4*i+2])*pri_width
                decoded_box_height = np.exp(pri[0,1,4*i+3]*loc[0,4*i+3])*pri_height
            elif not variance_exist:
                decoded_box_center_x = pri_center_x + loc[0,4*i+0]*pri_width
                decoded_box_center_y = pri_center_y + loc[0,4*i+1]*pri_height
                decoded_box_width = np.exp(loc[0,4*i+2])*pri_width
                decoded_box_height = np.exp(loc[0,4*i+3])*pri_height

            decoded_box_x_min = decoded_box_center_x - decoded_box_width/2.0
            decoded_box_y_min = decoded_box_center_y - decoded_box_height/2.0
            decoded_box_x_max = decoded_box_center_x + decoded_box_width/2.0
            decoded_box_y_max = decoded_box_center_y + decoded_box_height/2.0

            decoded_boxs.append((decoded_box_x_min,decoded_box_y_min,decoded_box_x_max,decoded_box_y_max))
        return np.array(decoded_boxs)	

    def jaccard_overlap(b1,b2):
        # print(b1)
        # print(b2)
        if b1[2]<b2[0] or b1[3]<b2[1] or b1[0]>b2[2] or b1[1]>b2[3]:
            # print("jaccard_overlap :",0)
            return 0
        x_min = max(b1[0],b2[0])
        y_min = max(b1[1],b2[1])
        x_max = min(b1[2],b2[2])
        y_max = min(b1[3],b2[3])
        intersection_area = (x_max-x_min)*(y_max-y_min)
        union_area = (b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-intersection_area
        jaccard_overlap = intersection_area/union_area
        # print("jaccard overlap :",jaccard_overlap)
        
        return jaccard_overlap

    def apply_nms(label,decoded_boxs,conf):
        conf_thresh = 0.01
        confre = conf.reshape(2268,5)
        decoded_boxsre = decoded_boxs.reshape(2268,4)
        tmp_id_score=[]
        tmp_num = 0
        for i in range(2268):
            if confre[i,label]>conf_thresh:
                tmp_id_score.append([i,confre[i,label]])
                tmp_num += 1
        tmp_id_score.sort(key = lambda x:x[1],reverse = True)
        tmp_num = min(400,tmp_num)
        for i in range(tmp_num):
            if tmp_id_score[i][1]<=conf_thresh:
                continue
            for j in range(i+1,tmp_num):
                if tmp_id_score[j][1]<=conf_thresh:
                    continue
                if jaccard_overlap(decoded_boxsre[tmp_id_score[i][0],:],decoded_boxsre[tmp_id_score[j][0],:])>0.45:
                    tmp_id_score[j][1]=0
        out_boxs = []
        for i in range(tmp_num):
            if tmp_id_score[i][1]<=conf_thresh:
                continue
            out_boxs.append([label,tmp_id_score[i][0],tmp_id_score[i][1],decoded_boxsre[tmp_id_score[i][0],:]])
        return out_boxs

            
    def detection_out(conf,decoded_boxs):
        conf_thresh = 0.01
        all_boxs = {i:[] for i in range(5)}
        all_boxs_tmp = []
        for i in range(5):
            box_tmp = apply_nms(i,decoded_boxs,conf)
            if i == 0:
                continue
            all_boxs_tmp.extend(box_tmp)
        
        num_det = min(len(all_boxs_tmp),200)
        all_boxs_tmp.sort(key = lambda x:x[2],reverse = True)
        for i in range(num_det):
            if all_boxs_tmp[i][2]<=conf_thresh:
                continue
            for j in range(i+1,num_det):
                if all_boxs_tmp[j][2]<=conf_thresh:
                    continue
                if jaccard_overlap(all_boxs_tmp[i][3],all_boxs_tmp[j][3])>0.45:
                    all_boxs_tmp[j][2]=0
        for i in range(num_det):
            if all_boxs_tmp[i][2]<=conf_thresh:
                continue
            all_boxs[all_boxs_tmp[i][0]].append(all_boxs_tmp[i][1:])
            
        return all_boxs
    decoded_boxs = decode_all_boxs(pri,loc)

    all_boxs = detection_out(conf,decoded_boxs)

    # img = cv2.imread(image_path)
    
    for class_id,boxs_info in all_boxs.items():
        for box_info in boxs_info:
            result = {}
            result['class_id'] = int(class_id)
            result['score'] = round(box_info[1],3)
            result['x_min'] = round(300*box_info[2][0],1)
            result['y_min'] = round(300*box_info[2][1],1)
            result['x_max'] = round(300*box_info[2][2],1)
            result['y_max'] = round(300*box_info[2][3],1)
            results[image_name].append(result)

            # p_left_top = (int(result['x_min']),int(result['y_min']))
            # p_right_bottom = (int(result['x_max']),int(result['y_max']))
            # cv2.rectangle(img,p_left_top,p_right_bottom,(255,255,0),2)
            
    # cv2.imshow('',img)
    # k = cv2.waitKey(1)

    num = num+1
    print('finish  {}.......'.format(num))
    
yaml.dump(results,fp)	
fp.close()
end = time.time()
time_consumption = end-start
print("time consumption: ",time_consumption)
