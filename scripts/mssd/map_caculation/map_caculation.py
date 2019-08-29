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

def caculate_ap(rec,prec):
    mrec = np.concatenate(([0],rec,[1]))
    mprec = np.concatenate(([0],prec,[0]))

    for i in range(mprec.size-1,0,-1):
        mprec[i-1]=np.maximum(mprec[i-1],mprec[i])
    
    i = np.where(mrec[1:]!=mrec[:-1])[0]

    ap = np.sum((mrec[i+1]-mrec[i])*mprec[i+1])

    return ap

if __name__ == '__main__':
    classnum = 5
    gt_root = '/home/huangrq/DataSet/testdata/annotations/'
    ovthresh = 0.5
    use_difficult = 0
    labels = ['background','car','tricycle','pedestrian','rider']

    fp = open('mssd_with_det_result.yaml')
    dt_out = yaml.load(fp)
    ##dt_out is a dict,key is picture name,value include the boxinfo detected in the image
    fp.close()
    class_det = {i:[] for i in labels}
    gts = {}
    gt_box_num_class = {i:0 for i in labels}
    for img_name,one_img_detinfo in dt_out.items():
        gt_path = gt_root+img_name.replace('.jpg','.xml')
        gts[img_name],one_gt_box_num=get_gt(gt_path)
        for key,value in one_gt_box_num.items():
            gt_box_num_class[key] = gt_box_num_class[key]+value
        if one_img_detinfo is None:
            continue
        for det_info in one_img_detinfo:
            class_det[labels[det_info['class_id']]].append([img_name,det_info['score'],det_info['x_min'],det_info['y_min'],det_info['x_max'],det_info['y_max']])

    fp = open('class_det.yaml','w')
    yaml.dump(class_det,fp)
    fp.close()

    MAP = {}
    for class_id,class_results in class_det.items():
        if len(class_results) == 0:
            continue
        class_results.sort(key = lambda x:x[1],reverse = True)
        nd = len(class_results)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for i,one_det_box in enumerate(class_results):
            
            gt = gts[one_det_box[0]][class_id]
            if len(gt) == 0:
                fp[i] = 1
            else:
                gt_tmp = []
                for gt_box in gt:
                    gt_tmp.append(gt_box['bbox'])
                gt_boxs_array = np.array(gt_tmp)     #shape n x 4
                det_box_array = np.array([one_det_box[2],one_det_box[3],one_det_box[4],one_det_box[5]])
                ix_min = np.maximum(gt_boxs_array[:,0],det_box_array[0])
                iy_min = np.maximum(gt_boxs_array[:,1],det_box_array[1])
                ix_max = np.minimum(gt_boxs_array[:,2],det_box_array[2])
                iy_max = np.minimum(gt_boxs_array[:,3],det_box_array[3])
                iw = np.maximum(ix_max-ix_min+1,0)
                ih = np.maximum(iy_max-iy_min+1,0)
                intersection_area = iw*ih
                union_area = (det_box_array[2]-det_box_array[0]+1)*(det_box_array[3]-det_box_array[1]+1)+(gt_boxs_array[:,2]-gt_boxs_array[:,0]+1)*(gt_boxs_array[:,3]-gt_boxs_array[:,1]+1)-intersection_area   

                overlaps =  intersection_area/1.0/union_area
                ovmax = np.max(overlaps)
                indexmax = np.argmax(overlaps)

                if ovmax>ovthresh:
                    if gt[indexmax]['difficult']==0 or use_difficult==1:
                        if gt[indexmax]['det'] == 0:
                            gt[indexmax]['det'] = 1
                            tp[i] = 1
                        else:
                            fp[i] = 1
                else:
                    fp[i] = 1
        
        gt_box_num = gt_box_num_class[class_id]
        if gt_box_num == 0:
            AP = None
        else:
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)

            rec = tp/float(gt_box_num)
            prec = tp/np.maximum(fp+tp,np.finfo(float).eps)
            AP = caculate_ap(rec,prec)
            MAP[class_id] = AP
    print(MAP)
    print('end')