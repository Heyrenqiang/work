import glob

path = '/home/huangrq/DataSet/testdata/image_300x300/*'
filenames = []

fp = open("files.txt",'w+')
for item in glob.glob(path):
    item=item.replace('/home/huangrq/DataSet/testdata/image_300x300/','')
    filenames.append(item)
    fp.write(item+'\n')

fp.close()

