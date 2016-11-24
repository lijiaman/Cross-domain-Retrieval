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



def load_batchsize_images(batch_size=32):
	street_path_batch = []
        shop_path_batch = []
        nopair_path_batch = []
	street_batch = []
	shop_batch = []
        nopair_batch = []
        y_street_batch = []
        y_shop_batch = []
        y_nopair_batch = []
	with open(tmp+'toy64_noshare_train.txt') as f:
	#with open(tmp+'list_train_triplet_category.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[3]) and os.path.isfile(img_path+line[5]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[2],10)-1)
                            shop_path_batch.append(img_path+line[3])
                            y_shop_batch.append(string.atoi(line[4],10)-1)
                            nopair_path_batch.append(img_path+line[5])
                            y_nopair_batch.append(string.atoi(line[6],10)-1)
    	
	f.close()
	for street_path in street_path_batch:
                #street_tmp = load_image(street_path)
                #print street_tmp.shape
                #print "img path:"
                #print street_path
                #assert(street_tmp.shape == [227,227,3])
		street_batch.append(load_image(street_path))
        for shop_path in shop_path_batch:
                shop_batch.append(load_image(shop_path))
        for nopair_path in nopair_path_batch:
                nopair_batch.append(load_image(nopair_path))
	return np.asarray(street_batch), np.asarray(shop_batch), np.asarray(nopair_batch), np.asarray(y_street_batch), np.asarray(y_shop_batch), np.asarray(y_nopair_batch)

def load_val_images(batch_size=32):
	street_path_batch = []
        shop_path_batch = []
        nopair_path_batch = []
	street_batch = []
	shop_batch = []
        nopair_batch = []
        y_street_batch = []
        y_shop_batch = []
        y_nopair_batch = []
	with open(tmp+'toy64_noshare_val.txt', 'rb') as f:
	#with open(tmp+'list_val_triplet_category.txt', 'rb') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[3]) and os.path.isfile(img_path+line[5]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[2],10)-1)
                            shop_path_batch.append(img_path+line[3])
                            y_shop_batch.append(string.atoi(line[4],10)-1)
                            nopair_path_batch.append(img_path+line[5])
                            y_nopair_batch.append(string.atoi(line[6],10)-1)
    	
	f.close()
	for street_path in street_path_batch:
                #street_tmp = load_image(street_path)
                #print street_tmp.shape
                #print "img path:"
                #print street_path
                #assert(street_tmp.shape == [227,227,3])
		street_batch.append(load_image(street_path))
        for shop_path in shop_path_batch:
                shop_batch.append(load_image(shop_path))
        for nopair_path in nopair_path_batch:
                nopair_batch.append(load_image(nopair_path))
	return np.asarray(street_batch), np.asarray(shop_batch), np.asarray(nopair_batch), np.asarray(y_street_batch), np.asarray(y_shop_batch), np.asarray(y_nopair_batch)

def load_test_images(test_batch_size=32):
	street_path_batch = []
        shop_path_batch = []
        nopair_path_batch = []
	street_batch = []
	shop_batch = []
        nopair_batch = []
        y_street_batch = []
        y_shop_batch = []
        y_nopair_batch = []
	with open(tmp+'toy64_noshare_test.txt', 'rb') as f:
        #with open(tmp+'list_test_triplet_category.txt') as f:
		lines = random.sample(f.readlines(),test_batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[3]) and os.path.isfile(img_path+line[5]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[2],10)-1)
                            shop_path_batch.append(img_path+line[3])
                            y_shop_batch.append(string.atoi(line[4],10)-1)
                            nopair_path_batch.append(img_path+line[5])
                            y_nopair_batch.append(string.atoi(line[6],10)-1)
    	
	f.close()
	for street_path in street_path_batch:
                #street_tmp = load_image(street_path)
                #print street_tmp.shape
                #print "img path:"
                #print street_path
                #assert(street_tmp.shape == [227,227,3])
		street_batch.append(load_image(street_path))
        for shop_path in shop_path_batch:
                shop_batch.append(load_image(shop_path))
        for nopair_path in nopair_path_batch:
                nopair_batch.append(load_image(nopair_path))
	return np.asarray(street_batch), np.asarray(shop_batch), np.asarray(nopair_batch), np.asarray(y_street_batch), np.asarray(y_shop_batch), np.asarray(y_nopair_batch)

def load_share_train(batch_size=32):
	street_path_batch = []
        shop_path_batch = []
	street_batch = []
	shop_batch = []
        y_street_batch = []
        y_shop_batch = []
        if_pair_batch = []
	#with open(tmp+'split_triplet_train.txt') as f:
	with open(tmp+'toy64_share_nin_train.txt') as f:
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

def load_share_val(batch_size=32):
	street_path_batch = []
        shop_path_batch = []
	street_batch = []
	shop_batch = []
        y_street_batch = []
        y_shop_batch = []
        if_pair_batch = []
	#with open(tmp+'split_triplet_val.txt') as f:
	with open(tmp+'toy64_share_nin_val.txt') as f:
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

    
def load_share_test(batch_size=32):
	street_path_batch = []
        shop_path_batch = []
	street_batch = []
	shop_batch = []
        y_street_batch = []
        y_shop_batch = []
        if_pair_batch = []
	#with open(tmp+'split_triplet_test.txt') as f:
	with open(tmp+'toy64_share_nin_test.txt') as f:
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

