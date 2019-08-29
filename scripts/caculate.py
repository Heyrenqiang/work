#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
'''
caculate the accuracy of VehicleHead
'''
from functools import reduce
t_start=time.time()

num_of_color=11
num_of_model=10
num_of_type=1159
num_of_data=970

filename_label = "/home/huangrq/vivworkspace/VehicleHead/test_list_w.txt"
filename_result = "/home/huangrq/vivworkspace/VehicleHead/output_list.txt"
fileout1="/home/huangrq/vivworkspace/VehicleHead/not_match_color_list.txt"
fileout2="/home/huangrq/vivworkspace/VehicleHead/not_match_model_list.txt"
fileout3="/home/huangrq/vivworkspace/VehicleHead/not_match_type_list.txt"
fp1 = open(filename_label)
fp2 = open(filename_result)
linelist1 = fp1.readlines()
linelist2 = fp2.readlines()
fp1.close()
fp2.close()
linelist1[0].split(" ")
#print(linelist1[0])
color_match_list = map(lambda x,y:x.strip().split(" ")[1]==y.split(" ")[1][6:] ,linelist1,linelist2)
model_match_list = map(lambda x,y:x.strip().split(" ")[2]==y.split(" ")[3][6:] ,linelist1,linelist2)
type_match_list = map(lambda x,y:x.strip().split(" ")[3]==y.split(" ")[5][5:] ,linelist1,linelist2)
color_match_list=list(color_match_list)
model_match_list=list(model_match_list)
type_match_list=list(type_match_list)

str=linelist2[0].split(" ")[5]

color_match_num = reduce(lambda x,y:x+y,color_match_list)
model_match_num = reduce(lambda x,y:x+y,model_match_list)
type_match_num = reduce(lambda x,y:x+y,type_match_list)

print(color_match_num,model_match_num,type_match_num)
print(color_match_num/num_of_data,model_match_num/num_of_data,type_match_num/num_of_data)

not_match_color_list=[]
index=0
for item in color_match_list:
    while not item:
        not_match_color_list.append(linelist1[index].strip('\n')+"  |  "+linelist2[index])
        break
    index = index + 1
not_match_model_list=[]
index =0
for item in model_match_list:
    while not item:
        not_match_model_list.append(linelist1[index].strip('\n')+"  |  "+linelist2[index])
        break
    index = index + 1
not_match_type_list=[]
index =0
for item in type_match_list:
    while not item:
        not_match_type_list.append(linelist1[index].strip('\n')+"  |  "+linelist2[index])
        break
    index = index + 1

color_n_match=[[] for i in range(11)]
model_n_match=[[] for i in range(10)]
type_n_match=[[] for i in range(1159)]
color_n_match_dict=[{j:0 for j in range(11)} for i in range(11)]
model_n_match_dict=[{j:0 for j in range(10)} for i in range(10)]
type_n_match_dict=[{j:0 for j in range(1159)} for i in range(1159)]
for item in not_match_color_list:
    if int(item.split()[1])==0:
        color_n_match[0].append(int(item.split()[6][6:]))
        color_n_match_dict[0][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==1:
        color_n_match[1].append(int(item.split()[6][6:]))
        color_n_match_dict[1][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==2:
        color_n_match[2].append(int(item.split()[6][6:]))
        color_n_match_dict[2][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==3:
        color_n_match[3].append(int(item.split()[6][6:]))
        color_n_match_dict[3][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==4:
        color_n_match[4].append(int(item.split()[6][6:]))
        color_n_match_dict[4][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==5:
        color_n_match[5].append(int(item.split()[6][6:]))
        color_n_match_dict[5][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==6:
        color_n_match[6].append(int(item.split()[6][6:]))
        color_n_match_dict[6][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==7:
        color_n_match[7].append(int(item.split()[6][6:]))
        color_n_match_dict[7][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==8:
        color_n_match[8].append(int(item.split()[6][6:]))
        color_n_match_dict[8][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==9:
        color_n_match[9].append(int(item.split()[6][6:]))
        color_n_match_dict[9][int(item.split()[6][6:])]+=1
    elif int(item.split()[1])==10:
        color_n_match[10].append(int(item.split()[6][6:]))
        color_n_match_dict[10][int(item.split()[6][6:])]+=1
for item in not_match_model_list:
    if int(item.split()[2])==0:
        model_n_match[0].append(int(item.split()[8][6:]))
        model_n_match_dict[0][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==1:
        model_n_match[1].append(int(item.split()[8][6:]))
        model_n_match_dict[1][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==2:
        model_n_match[2].append(int(item.split()[8][6:]))
        model_n_match_dict[2][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==3:
        model_n_match[3].append(int(item.split()[8][6:]))
        model_n_match_dict[3][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==4:
        model_n_match[4].append(int(item.split()[8][6:]))
        model_n_match_dict[4][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==5:
        model_n_match[5].append(int(item.split()[8][6:]))
        model_n_match_dict[5][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==6:
        model_n_match[6].append(int(item.split()[8][6:]))
        model_n_match_dict[6][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==7:
        model_n_match[7].append(int(item.split()[8][6:]))
        model_n_match_dict[7][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==8:
        model_n_match[8].append(int(item.split()[8][6:]))
        model_n_match_dict[8][int(item.split()[8][6:])]+=1
    elif int(item.split()[2])==9:
        model_n_match[9].append(int(item.split()[8][6:]))
        model_n_match_dict[9][int(item.split()[8][6:])]+=1
print(color_n_match) 
print(model_n_match)  
print(color_n_match_dict)
print(model_n_match_dict)
t_end = time.time()
print(t_end-t_start)
