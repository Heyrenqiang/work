#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
import cv2
import caffe
caffe.set_mode_cpu()

model_def = '/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/VehicleBack.prototxt'
model_weights = '/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/VehicleBack.caffemodel' 
net = caffe.Net(model_def,model_weights,caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})


transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', np.array([127.5,127.5,127.5]))            #
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[-1,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR



print("###############################################################")
filename_label = "/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/test_list_w.txt"
file_out="/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/AC_oncaffe_out.txt"
fp = open(filename_label)
fpout = open(file_out,'w')
filedata=fp.readlines()
fp.close()
filepath = list(map(lambda x:x.strip().split()[0],filedata))
label_color=list(map(lambda x:int(x.strip().split()[1]),filedata))
label_model=list(map(lambda x:int(x.strip().split()[2]),filedata))
label_type=list(map(lambda x:int(x.strip().split()[3]),filedata))
print(label_color)

model_statistic=[0 for i in range(10)]
for item in label_model:
    for i in range(10):
        if i==item:
            model_statistic[i] += 1
            break
print(label_model)
color_statistic=[0 for i in range(11)]
for item in label_color:
    for i in range(11):
        if i==item:
            color_statistic[i] += 1
            break
print(color_statistic,'\n',model_statistic)
model,color,types=[],[],[]

for item in filepath:

    image=caffe.io.load_image('/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/'+item)
    transformed_image=transformer.preprocess('data',image)
    transformed_image = transformed_image/127.5

    net.blobs['data'].data[...] = transformed_image
    out_put = net.forward()

    
    model.append(out_put['prob_model'].argmax())
    color.append(out_put['prob_color'].argmax())
    types.append(out_put['prob_type'].argmax())



out_result=list(map(lambda x,y,z,w:w+' model: '+str(x)+' color: '+str(y)+'  type: '+str(z)+'\n',model,color,types,filepath))
fpout.writelines(out_result)

fpout.close()



'''
compare label and predict
'''

from functools import reduce
filename_label = "/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/test_list_w.txt"
file_out="/home/huangrq/vivworkspace/VehicleBack_base_batchsize1_test/AC_oncaffe_out.txt"
label_fp = open(filename_label)
predict_fp = open(file_out)
label_filedata=label_fp.readlines()
predict_filedata=predict_fp.readlines()
label_fp.close()
predict_fp.close()

label_filepath = list(map(lambda x:x.strip().split()[0],label_filedata))
label_color=list(map(lambda x:int(x.strip().split()[1]),label_filedata))
label_model=list(map(lambda x:int(x.strip().split()[2]),label_filedata))
label_type=list(map(lambda x:int(x.strip().split()[3]),label_filedata))

predict_filepath = list(map(lambda x:x.strip().split()[0],predict_filedata))
predict_color=list(map(lambda x:int(x.strip().split()[4]),predict_filedata))
predict_model=list(map(lambda x:int(x.strip().split()[6]),predict_filedata))
predict_type=list(map(lambda x:int(x.strip().split()[2]),predict_filedata))

print(predict_type[4])
print(label_type[4])

color_match_list=list(map(lambda x,y:x==y,label_color,predict_color))
model_match_list=list(map(lambda x,y:x==y,label_model,predict_model))
type_match_list=list(map(lambda x,y:x==y,label_type,predict_type))

color_match_num = reduce(lambda x,y:x+y,color_match_list)
model_match_num = reduce(lambda x,y:x+y,model_match_list)
type_match_num = reduce(lambda x,y:x+y,type_match_list)

print(color_match_num,model_match_num,type_match_num)