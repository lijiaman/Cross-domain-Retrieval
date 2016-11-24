import numpy as np
import json
import time
street_file = '/ais/gobi4/fashion/retrieval/total_bbox_street_features.json'
shop_file = '/ais/gobi4/fashion/retrieval/bbox_test_gallery.json'
shop_total = 47384
street_total = 200
def build_similarity():
	street_mat = []
	shop_mat = []
	with open(street_file, 'rb') as street:
	    start_time = time.time()
	    street_data = street.readlines()
	    for street_line in street_data:
		street_line = json.loads(street_line)
		cal_street = np.asarray(street_line['street_feature'])
		street_mat.append(cal_street)
	    print("Build street mat took:")
	    print time.time()-start_time
	    print("street_mat.shape:")
	    street_mat = np.asarray(street_mat)
	    street_mat = street_mat.reshape(street_mat.shape[0], -1)
	    print street_mat.shape
	street.close()        

	with open(shop_file, 'rb') as shop:
	    start_time = time.time()
	    shop_data = shop.readlines()
	    for shop_line in shop_data:
		shop_line = json.loads(shop_line)
		cal_shop = np.asarray(shop_line['shop_feature'])
		shop_mat.append(cal_shop)
	    print("Build shop mat took:")
	    print time.time()-start_time
	    print("shop_mat.shape:")
	    shop_mat = np.asarray(shop_mat)
	    shop_mat = shop_mat.reshape(shop_mat.shape[0], -1)
	    print shop_mat.shape
	shop.close()     

	similarity = np.dot(street_mat, shop_mat.T)

	similar_file = '/ais/gobi4/fashion/retrieval/bbox_similarity.npy'
	np.save(similar_file, similarity)

def get_topk_acc(k=20, street_num=47384):
    similar_file = '/ais/gobi4/fashion/retrieval/bbox_similarity.npy'
    simi_vec = np.load(similar_file)
    #print("similarity shape:")
    #print simi_vec.shape
    hit = 0
    acc =0
    index = np.argsort(-simi_vec)  
    topk = index[:, :k]
    with open(street_file, 'rb') as str_f:
            str_lines = str_f.readlines()
	    for i in xrange(street_num):
                start_time = time.time()
		single = topk[i]
	        line_i = str_lines[i]
                line_i = json.loads(line_i)	
                
		with open(shop_file, 'rb') as f:
                    lines = f.readlines()
		    for j in single:
                        #print("j")
                        #print j
                        line_j = lines[j]
			#line_j = f.readlines()[j]
			line_j = json.loads(line_j)
			if line_j['id'] == line_i['id']:
                            hit += 1
                            break
  
                f.close()
                print("One iteration took:")
                print time.time() - start_time
                print("{0}/{1}:hit={2}".format(i+1, street_num, hit))
    str_f.close()
    acc = float(hit) / street_num
    print "acc:"
    print acc 

build_similarity()
get_topk_acc(k=20)
