import numpy as np 
import random
import linecache
import os
import retrieval_input
tmp = '/ais/gobi4/fashion/data/Cross-domain-Retrieval/'
def split_data():
	with open(tmp+'list_val_pairs.txt','a') as w_f:
		with open(tmp+'Eval/list_eval_partition.txt') as f:
			data = f.readlines()
			line_cnt = 0
			for line in data:
				line = line.split()
        		        line_cnt += 1
        		        if line_cnt > 2:
        			        if line[3] == 'val':
                                            if os.path.isfile(tmp+line[0]) and os.path.isfile(tmp+line[1]):
                                                street_img = retrieval_input.load_image(tmp+line[0])
                                                shop_img = retrieval_input.load_image(tmp+line[1])
                                                if street_img.size == 227*227*3 and shop_img.size == 227*227*3:
        				            w_f.write(line[0]+'\t'+line[1]+'\t'+line[2]+'\n')
                f.close()
        w_f.close()



def make_triplet(num_lines):
	with open(tmp+'list_val_triplet.txt','a') as w_f:
		with open(tmp+'list_val_pairs.txt') as f:
			data = f.readlines()
			for line in data:
				line = line.split()
                                nopair_index = np.random.choice(num_lines,1)
                                third = linecache.getline(tmp+'list_val_pairs.txt',nopair_index+1)
                                not_pair = third.split()
                                #print not_pair
                                #print "not_pair[2]"
                                #print not_pair[2]        
			        #third = random.sample(f.readlines(),1)
			        #not_pair = third[0].split()
			        while not_pair[2] == line[2]:
			    	    #third = random.sample(f.readlines(),1)
			    	    #not_pair = third[0].split()
                                    nopair_index = np.random.choice(num_lines,1)
                                    third = linecache.getline(tmp+'list_val_pairs.txt',nopair_index+1)
                                    not_pair = third.split()
                                    #print "not_pair:"
                                    #print not_pair
			        w_f.write(line[2]+'\t'+line[0]+'\t'+line[1]+'\t'+not_pair[1]+'\n')
		f.close()   
        w_f.close()

def get_dict():
    dict = {}
    with open(tmp+'category_23.txt') as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            dict[line[1]] = line[0]
    f.close()
    return dict

def get_category(dict):
    with open(tmp+'list_val_triplet_category.txt','a') as w_f:
        with open(tmp+'list_val_triplet.txt') as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                for item in dict:
                    if item in line[1]:
                        w_f.write(line[0]+'\t'+line[1]+'\t'+dict[item]+'\t'+line[2]+'\t'+dict[item]+'\t')
                for item in dict:
                    if item in line[3]:
                        w_f.write(line[3]+'\t'+dict[item]+'\n')
                        
        f.close()
    w_f.close()         

def get_toy_data(dataset_size, toy_file, src_file):
    with open(tmp+toy_file,'a') as w_f:
        with open(tmp+src_file) as f:
            lines = random.sample(f.readlines(),dataset_size)
            for line in lines:
                w_f.write(line)
        f.close()
    w_f.close()

def split_triplet(dest_file, src_file):
    with open(tmp+dest_file, 'a') as w_f:
        with open(tmp+src_file) as f:
            data = f.readlines()
            for line in data:
                line = line.split()
                pair = 1
                nopair = 0
                w_f.write(str(pair)+'\t'+line[1]+'\t'+line[3]+'\t'+line[2]+'\t'+line[4]+'\n')
                w_f.write(str(nopair)+'\t'+line[1]+'\t'+line[5]+'\t'+line[2]+'\t'+line[6]+'\n')
        f.close()
    w_f.close() 


#split_triplet('split_triplet_train.txt', 'list_train_triplet_category.txt')
#get_toy_data(64, 'toy64_share_nin_train.txt', 'split_triplet_train.txt')

#split_triplet('split_triplet_val.txt', 'list_val_triplet_category.txt')
#get_toy_data(64, 'toy64_share_nin_val.txt', 'split_triplet_val.txt')

#split_triplet('split_triplet_test.txt', 'list_test_triplet_category.txt')
#get_toy_data(64, 'toy64_share_nin_test.txt', 'split_triplet_test.txt')
#get_toy_data(20)
get_toy_data(64, 'toy64_noshare_train.txt', 'list_train_triplet_category.txt')
get_toy_data(64, 'toy64_noshare_val.txt', 'list_val_triplet_category.txt')
get_toy_data(64, 'toy64_noshare_test.txt', 'list_test_triplet_category.txt')
#dict = get_dict()
#get_category(dict)
#def triplet_add_category():
#split_data()
#num_lines = sum(1 for line in open(tmp+'list_val_pairs.txt'))
#print "num_lines:"
#print num_lines

#make_triplet(num_lines)
