import numpy as np
import json
import time
#street_file = '/ais/gobi4/fashion/retrieval/part_street_features.json'
#shop_file = '/ais/gobi4/fashion/retrieval/test_gallery.json'
#street_file = '/ais/gobi4/fashion/retrieval/toy4_darn_street_features.json'
#shop_file = '/ais/gobi4/fashion/retrieval/toy4_darn_test_gallery.json'
street_file = '/ais/gobi4/fashion/retrieval/toy4_share_street_features.json'
shop_file = '/ais/gobi4/fashion/retrieval/toy4_share_test_gallery.json'
k = 1
acc = 0
#shop_total = 47384
street_total = 24
with open(street_file, 'rb') as street:
    start_time = time.time()
    street_data = street.readlines()
    for street_line in street_data:
        street_line = json.loads(street_line)
        cal_street = np.asarray(street_line['street_feature'])
        distance = []
        with open(shop_file, 'rb') as shop:
            shop_data = shop.readlines()
            for shop_line in shop_data:
                shop_line = json.loads(shop_line)
                cal_shop = np.asarray(shop_line['shop_feature'])
                distance.append( np.sum((cal_shop-cal_street)**2))
        shop.close()
        index = np.argsort(distance)
        print("index:")
        print index
        topk_index = index[:k]
        print("topk index:")
        print topk_index
        cnt = 0
        with open(shop_file, 'rb') as shop:
            shop_data = shop.readlines()
            for i in topk_index:
                line = shop_data[i]
                select = json.loads(line)
                if select['id'] == street_line['id']:
                    cnt = 1
        shop.close()
        
        #print("hit = {0}".format(acc))
        acc += float(cnt)
        print("hit = {0}".format(acc))


    acc /= street_total
    print acc
    print time.time()-start_time
street.close()
