#!/usr/bin/env python
#coding:utf8
from itertools import groupby
from operator import itemgetter
import numpy as np
import random
import pdb
train_data = "../data/ml-1m/ratings.dat"
hidden_vector_size = 100
alpha = 0.01

def MatrixFactorization():
    data = [map(lambda y :int(y),x.strip().split("::")[:3]) for x in open(train_data)]
    uniq_user = []
    for key,items in groupby(data,itemgetter(0)):
        temp = [key,[x[1:] for x in items]]
        uniq_user.append(temp)
    users_num = len(uniq_user)
    uniq_movie = set([m[1] for m in data])
    user_param_dict = {}
    for k,v in uniq_user:
        user_param_dict[k] = np.random.randn(hidden_vector_size) 
    
    movie_param_dict = {}
    for movieid in uniq_movie:
        movie_param_dict[movieid] = np.random.randn(hidden_vector_size) 
        
    print users_num
    #SGD
    for loop in range(int(users_num * 60)):
        index = random.randint(0,users_num - 1)
        userid,rates = uniq_user[index]
        w0 = user_param_dict[userid]
        w0_g = np.zeros(hidden_vector_size)
        #pdb.set_trace()
        for movieid,target in rates:
            w1 = movie_param_dict[movieid]  
            g = alpha * (np.dot(w0,w1) - target) 
            movie_param_dict[movieid] -= g * w0
            w0_g += g * w1
        user_param_dict[userid] -= w0_g /len(rates) 
        if loop % 500 == 0:
            print loop
    totoal_error = 0
    total_num = 0
    for userid,rates in uniq_user:
        w0 = user_param_dict[userid]
        for movieid,target in rates:
            w1 = movie_param_dict[movieid]  
            score = np.dot(w0,w1)
            totoal_error += np.abs((score - target))
            total_num += 1
    print "total sqrt error:",totoal_error / total_num

if __name__ == '__main__':
    MatrixFactorization()








