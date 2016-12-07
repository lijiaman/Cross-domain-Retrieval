import numpy as np 
import string
import random
import linecache
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



def load_batchsize_images(batch_size=32,mode='train'):
	street_path_batch = []
        shop_path_batch = []
        nopair_path_batch = []
	street_batch = []
	shop_batch = []
        nopair_batch = []
        y_street_batch = []
        y_shop_batch = []
        y_nopair_batch = []
        #train_file = 'toy4_noshare_train_category.txt'
        if mode == 'train':
            train_file = 'train_pairs_category.txt'
            num_lines = 98163
        elif mode == 'val':
            train_file = 'val_pairs_category.txt'
            num_lines = 48571
        elif mode == 'test':
            train_file = 'test_pairs_category.txt'
            num_lines = 47384
        else:
            print("mode is not valid!")

	with open(tmp+train_file) as f:
	#with open(tmp+'list_train_triplet_category.txt') as f:
		lines = random.sample(f.readlines(),batch_size)
    	#print lines
    	        for line in lines:
    		        line = line.split()
                        nopair_index = np.random.choice(num_lines, 1)
                        third = linecache.getline(tmp+train_file, nopair_index+1)
                        not_pair = third.split()

                        while not_pair[0] == line[0]:
                            nopair_index = np.random.choice(num_lines, 1)
                            third = linecache.getline(tmp+train_file, nopair_index+1)
                            not_pair = third.split()

                        if os.path.isfile(img_path+line[1]) and os.path.isfile(img_path+line[3]) and os.path.isfile(img_path+not_pair[3]):
    		            street_path_batch.append(img_path+line[1])
                            y_street_batch.append(string.atoi(line[2],10)-1)
                            shop_path_batch.append(img_path+line[3])
                            y_shop_batch.append(string.atoi(line[4],10)-1)
                            #print("not_pair:{0}".format(not_pair[3]))
                            nopair_path_batch.append(img_path+not_pair[3])
                            y_nopair_batch.append(string.atoi(not_pair[4],10)-1)
    	
	f.close()
	for street_path in street_path_batch:
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
	with open(tmp+'toy64_share_train.txt') as f:
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
	with open(tmp+'toy64_share_val.txt') as f:
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
	with open(tmp+'toy64_share_test.txt') as f:
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

