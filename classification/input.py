import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
import sys
sys.path.append('/ais/gobi4/fashion/data/Category-Attribute-Prediction/')
img_path = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
tmp = '/ais/gobi4/fashion/data/Category-Attribute-Prediction/'
#Build a dictionary for img path and category ID for the convenience of train,test,val split.
def get_dict():
	dic = {}
	with open(tmp+'list_category_img.txt') as f:
		data = f.readlines()
		line_cnt = 0
		for line in data:
			line_cnt += 1
			line = line.split()
			if line_cnt > 2:
                                if os.path.isfile(img_path+line[0]):
				    dic[line[0]] = line[1]
	f.close()
	return dic
def get_img_bbox_dict():
    dic = {}
    with open(tmp+'Anno/list_bbox.txt') as f:
        data = f.readlines()
        line_cnt = 0
        for line in data:
            line_cnt += 1
            line = line.split()
            if line_cnt > 2:
                if os.path.isfile(img_path+line[0]):
                    dic[(line[0],1)] = line[1]
                    dic[(line[0],2)] = line[2]
                    dic[(line[0],3)] = line[3]
                    dic[(line[0],4)] = line[4]
    f.close()
    return dic
#Used to split the training,testing and validation data.
def split_status(dic):
	with open(tmp+'list_category_test.txt','a') as w_f:
		with open(tmp+'list_test.txt') as f:
			val_data = f.readlines()
			for val_line in val_data:
				val_line = val_line.split()
                                if os.path.isfile(img_path+val_line[0]):
				    w_f.write(val_line[0]+'\t'+dic[val_line[0]]+'\n')
		f.close()
	w_f.close()

def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (227, 227))
    return resized_img

def load_batchsize_images(batch_size=128):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
	with open(tmp+'list_category_val.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for path in path_batch:
		x_batch.append(load_image(path))
	return np.asarray(x_batch), np.asarray(y_batch)
def get_bbox_file(dict):
    with open(tmp+'bbox_category_test.txt','a') as w_f:
        with open(tmp+'list_category_test.txt') as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                w_f.write(line[0]+'\t'+line[1]+'\t'+dict[(line[0],1)]+'\t'+dict[(line[0],2)]+'\t'+dict[(line[0],3)]+'\t'+dict[(line[0],4)]+'\n')
        f.close()
    w_f.close()
                
def load_bbox_image(path, x1, y1, x2, y2):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    #short_edge = min(img.shape[:2])
    #yy = int((img.shape[0] - short_edge) / 2)
    #xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[y1: y2, x1: x2]
    resized_img = skimage.transform.resize(crop_img, (227, 227))
    return resized_img

    
def load_batchsize_bbox_images(batch_size=128):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
        x1_batch = []
        y1_batch = []
        x2_batch = []
        y2_batch = []
	with open(tmp+'bbox_category_train.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
                        x1_batch.append(string.atoi(line[2],10))
                        y1_batch.append(string.atoi(line[3],10))
                        x2_batch.append(string.atoi(line[4],10))
                        y2_batch.append(string.atoi(line[5],10))
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for i in xrange(batch_size):
            x_batch.append(load_bbox_image(path_batch[i], x1_batch[i], y1_batch[i], x2_batch[i], y2_batch[i]))
        f.close()
        
        return x_batch, y_batch

def load_val_bbox_images(batch_size=128):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
        x1_batch = []
        y1_batch = []
        x2_batch = []
        y2_batch = []
	with open(tmp+'bbox_category_val.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
                        x1_batch.append(string.atoi(line[2],10))
                        y1_batch.append(string.atoi(line[3],10))
                        x2_batch.append(string.atoi(line[4],10))
                        y2_batch.append(string.atoi(line[5],10))
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for i in xrange(batch_size):
            x_batch.append(load_bbox_image(path_batch[i], x1_batch[i], y1_batch[i], x2_batch[i], y2_batch[i]))
        f.close()
        
        return x_batch, y_batch
def load_test_bbox_images(batch_size=128):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
        x1_batch = []
        y1_batch = []
        x2_batch = []
        y2_batch = []
	with open(tmp+'bbox_category_test.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
                        x1_batch.append(string.atoi(line[2],10))
                        y1_batch.append(string.atoi(line[3],10))
                        x2_batch.append(string.atoi(line[4],10))
                        y2_batch.append(string.atoi(line[5],10))
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for i in xrange(batch_size):
            x_batch.append(load_bbox_image(path_batch[i], x1_batch[i], y1_batch[i], x2_batch[i], y2_batch[i]))
        f.close()
        
        return x_batch, y_batch

def load_all_images():
        path_batch = []
        x_batch = []
        y_batch = []
        y_index = []
        with open(tmp+'list_category_train.txt') as f:
                lines = f.readlines()
                for line in lines:
                        line = line.split()
                        path_batch.append(img_path+line[0])
                        y_batch.append(string.atoi(line[1],10))
                        y_index.append(string.atoi(line[1],10)-1)
        f.close()
        category_num = 50
        y_batch_f = np.zeros([len(y_batch),category_num])
        y_batch_f[xrange(len(y_batch)),y_index] = 1
        for path in path_batch:
                x_batch.append(load_image(path))
        return np.asarray(x_batch), np.asarray(y_batch_f)

def get_toy_dataset(dataset_size=200):
        with open(tmp+'list_toy_train_200.txt','a') as w_f:
                with open(tmp+'list_category_train.txt') as f:
                        lines = random.sample(f.readlines(),dataset_size)
                        for line in lines:
                                w_f.write(line)
                                #print "20 line:"
                                #print line
                f.close()
        w_f.close()
                
def load_val_images(batch_size=128):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
	with open(tmp+'list_category_val.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([batch_size, category_num])
        y_batch_f[xrange(batch_size),y_index] = 1
	for path in path_batch:
		x_batch.append(load_image(path))
	return np.asarray(x_batch), np.asarray(y_batch)


def load_test_images(test_num=100):
	path_batch = []
	x_batch = []
	y_batch = []
        y_index = []
	with open(tmp+'list_category_test.txt') as f:
		lines = random.sample(f.readlines(),test_num)
    	#print lines
    	        for line in lines:
    		        line = line.split()
    		        path_batch.append(img_path+line[0])
    		        y_batch.append(string.atoi(line[1],10)-1)
                        y_index.append(string.atoi(line[1],10)-1)
    	
	f.close()
        category_num = 50
        y_batch_f = np.zeros([test_num, category_num])
        y_batch_f[xrange(test_num),y_index] = 1
	for path in path_batch:
		x_batch.append(load_image(path))
	return np.asarray(x_batch), np.asarray(y_batch)

	
#Only use once to get suitable file for data preprocessing.
#dic = get_dict()
#split_status(dic)
#x_batch, y_batch = load_batchsize_images()
#print "x_batch.shape:"
#print x_batch.shape#125 because some images are missed...
#print "y_batch.shape:"
#print y_batch.shape
#get_toy_dataset()
dict = get_img_bbox_dict()
get_bbox_file(dict)
