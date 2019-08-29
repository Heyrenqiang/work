import numpy as np 
import caffe

caffe.set_mode_cpu()
net = caffe.Net('/home/huangrq/task/scripts/caffe_inference/models/sfd.prototxt',
'/home/huangrq/task/scripts/caffe_inference/models/sfd.caffemodel',caffe.TEST)
net.save('/home/huangrq/task/scripts/caffe_inference/models/sfd.caffemodel')
