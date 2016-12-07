import numpy as np 
import string
import random
import skimage
import skimage.io
import skimage.transform
import os
import sys
sys.path.append('/ais/gobi4/fashion/data/Cross-domain-Retrieval/')
img_path = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
tmp = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'

def load_image(path, x1, y1, x2, y2):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    crop_img = img[y1: y2, x1: x2]
    resized_img = skimage.transform.resize(crop_img, (227, 227))
    return resized_img

def get_bbox_dict():
    dic = {}
    with open(tmp+'Anno/list_bbox_consumer2shop.txt','rb') as f:
        data = f.readlines()
        line_cnt = 0
        for line in data:
            line_cnt += 1
            line = line.split()
            if line_cnt > 2:
                if os.path.isfile(img_path+line[0]):
                    dic[(line[0], 1)] = line[3]#x1
                    dic[(line[0], 2)] = line[4]#y1
                    dic[(line[0], 3)] = line[5]#x2
                    dic[(line[0], 4)] = line[6]#y2
    f.close()
    return dic

def get_bbox_file(dic):
    with open(tmp+'bbox_test_triplet_category.txt', 'a') as w_f:
        with open(tmp+'list_test_triplet_category.txt', 'rb') as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                w_f.write(line[0]+'\t'+line[1]+'\t'+line[2]+'\t'+dic[(line[1],1)]+'\t'+dic[(line[1], 2)]+'\t'+dic[(line[1],3)]+'\t'+dic[(line[1],4)]+'\t'+line[3]+'\t'+line[4]+'\t'+dic[(line[3],1)]+'\t'+dic[(line[3],2)]+'\t'+dic[(line[3],3)]+'\t'+dic[(line[3],4)]+'\t'+line[5]+'\t'+line[6]+'\t'+dic[(line[5], 1)]+'\t'+dic[(line[5],2)]+'\t'+dic[(line[5],3)]+'\t'+dic[(line[5],4)]+'\n')
        f.close()
    w_f.close()

def get_share_bbox_file(dic):
    with open(tmp+'bbox_share_train_triplet_category.txt', 'a') as w_f:
        with open(tmp+'split_triplet_train.txt', 'rb') as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                w_f.write(line[0]+'\t'+line[1]+'\t'+dic[(line[1],1)]+'\t'+dic[(line[1], 2)]+'\t'+dic[(line[1],3)]+'\t'+dic[(line[1],4)]+'\t'+line[2]+'\t'+dic[(line[2],1)]+'\t'+dic[(line[2],2)]+'\t'+dic[(line[2],3)]+'\t'+dic[(line[2],4)]+'\t'+line[3]+'\t'+line[4]+'\n')
        f.close()
    w_f.close()


def load_bbox_batch_images(batch_size, datafile):
	street_path_batch = []
        shop_path_batch = []
        nopair_path_batch = []
	street_batch = []
	shop_batch = []
        nopair_batch = []
        y_street_batch = []
        y_shop_batch = []
        y_nopair_batch = []
        x1_street_batch = []
        x2_street_batch = []
        y1_street_batch = []
        y2_street_batch = []
        x1_shop_batch = []
        x2_shop_batch = []
        y1_shop_batch = []
        y2_shop_batch = []
        x1_nopair_batch = []
        x2_nopair_batch = []
        y1_nopair_batch = []
        y2_nopair_batch = []
	#with open(tmp+'toy64_noshare_train.txt') as f:
	with open(tmp+datafile) as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[7]) and os.path.isfile(img_path+line[13]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[2],10)-1)
                            x1_street_batch.append(string.atoi(line[3], 10))
                            y1_street_batch.append(string.atoi(line[4], 10))
                            x2_street_batch.append(string.atoi(line[5], 10))
                            y2_street_batch.append(string.atoi(line[6], 10))
                            shop_path_batch.append(img_path+line[7])
                            y_shop_batch.append(string.atoi(line[8],10)-1)
                            x1_shop_batch.append(string.atoi(line[9], 10))
                            y1_shop_batch.append(string.atoi(line[10], 10))
                            x2_shop_batch.append(string.atoi(line[11], 10))
                            y2_shop_batch.append(string.atoi(line[12], 10))
                            nopair_path_batch.append(img_path+line[13])
                            y_nopair_batch.append(string.atoi(line[14],10)-1)
                            x1_nopair_batch.append(string.atoi(line[15], 10))
                            y1_nopair_batch.append(string.atoi(line[16], 10))
                            x2_nopair_batch.append(string.atoi(line[17], 10))
                            y2_nopair_batch.append(string.atoi(line[18], 10))
    	
	f.close()
        for i in xrange(batch_size):
		street_batch.append(load_bbox_image(street_path[i], x1_street_batch[i], y1_street_batch[i],x2_street_batch[i], y2_street_batch[i]))
		shop_batch.append(load_bbox_image(shop_path[i], x1_shop_batch[i], y1_shop_batch[i],x2_shop_batch[i], y2_shop_batch[i]))
		nopair_batch.append(load_bbox_image(nopair_path[i], x1_nopair_batch[i], y1_nopair_batch[i],x2_nopair_batch[i], y2_nopair_batch[i]))
	return np.asarray(street_batch), np.asarray(shop_batch), np.asarray(nopair_batch), np.asarray(y_street_batch), np.asarray(y_shop_batch), np.asarray(y_nopair_batch)


def load_share_bbox_batch(batch_size,datafile):
	street_path_batch = []
        shop_path_batch = []
	street_batch = []
	shop_batch = []
        y_street_batch = []
        y_shop_batch = []
        if_pair_batch = []
        x1_street_batch = []
        x2_street_batch = []
        y1_street_batch = []
        y2_street_batch = []
        x1_shop_batch = []
        x2_shop_batch = []
        y1_shop_batch = []
        y2_shop_batch = []
	#with open(tmp+'split_triplet_train.txt') as f:
	with open(tmp+datafile) as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[2]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[3],10)-1)
                            shop_path_batch.append(img_path+line[2])
                            y_shop_batch.append(string.atoi(line[4],10)-1)
    	                    if_pair_batch.append(string.atoi(line[0],10))
        f.close()
	for street_path in street_path_batch:
		street_batch.append(load_image(street_path))
        for shop_path in shop_path_batch:
                shop_batch.append(load_image(shop_path))
	
        return np.asarray(street_batch), np.asarray(shop_batch), np.asarray(y_street_batch), np.asarray(y_shop_batch), np.asarray(if_pair_batch)


#dic = get_bbox_dict()
#get_bbox_file(dic)
