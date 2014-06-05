'''
Created on Jun 2, 2014

@author: Sean
'''

from math import sqrt
from sys import argv
import random

def add_vectors(a, b):
    '''Add vectors a and b '''
    #assert len(a) == len(b)
    #return [a[i]+b[i] for i in range(2)]
    return [a[i]+b[i] for i in range(len(a))]

def multiply_scalar_vector(alpha, vec):
    '''Multiply vector vec with scalar alpha '''
    return [alpha*f for f in vec]

def dot_product(a, b):
    '''Dot product of two vectors '''
    assert len(a) == len(b)
    return sum([a[i]*b[i] for i in range(len(a))])
    
def l2_norm(vec):
    '''Compute the L2 norm of the vector vec'''
    return sqrt(sum([vec[i]**2 for i in range(len(vec))]))

def verbose_classifier(filename, T, size):
    '''Read a training file and learn the parameters for each class'''
    training_file = open(filename)
    
    # Read the dimensions and the number of training points in each class
    D, nTrue, nFalse = map(int, training_file.readline().split())
    input = []
    
    count = 0 # Line and class counters
    for line in training_file:       
        x = list(map(float, line.split()))
        if count < nTrue:
            x.append(1)
        else:
            x.append(0)
        input.append(x)
        count+=1
    training_file.close()
    
    total_set = []
    for i in range(T):
        bootstrap_set = []
        for j in range(size):
            set = random.randint(0, nTrue+nFalse-1)
            temp = input[set]
            bootstrap_set.append(temp)
        total_set.append(bootstrap_set)
        
    W = []
    for i in range(len(total_set)):
        bootstrap = total_set[i]
        true_centroid = [0]*D
        false_centroid = [0]*D
        true_count = 0
        false_count = 0
        for j in range(size):
            temp = bootstrap[j]
            if temp[-1] == 1:
                true_centroid = add_vectors(true_centroid, temp)
                true_count+=1
            else:
                false_centroid = add_vectors(false_centroid, temp)
                false_count+=1
        true_centroid = multiply_scalar_vector(1.0/true_count, true_centroid)
        false_centroid = multiply_scalar_vector(1.0/false_count, false_centroid)
        w = add_vectors(true_centroid, multiply_scalar_vector(-1, false_centroid))
        t = (l2_norm(true_centroid)**2 - l2_norm(false_centroid)**2)/2.0
        w.append(t)
        W.append(w) 
    print(W)
        
    return W

def verbose_test_classifier(filename, W):
    testing_file = open(filename)
    
    D, nTrue, nFalse = map(int, testing_file.readline().split())
    tp, fp, tn, fn = 0,0,0,0
    count = 0
    for line in testing_file:
        x = list(map(float, line.split()))
        x.append(-1)
        vote = 0
        for i in range(len(W)):
            if dot_product(W[i], x) >= 0:
                vote += 1
            else:
                vote += -1
        
        if count < nTrue:
            actual_class = 0
        else:
            actual_class = 1
            
        if actual_class == 0:
            if vote >= 0:
                tp+=1
            else:
                fp+=1
        else:
            if vote >= 0:
                fn+=1
            else:
                tn+=1
                
    print("Positive examples:",end=' ')
    print(tp+fp)
    print("Negative examples:",end=' ')
            
    

if __name__ == '__main__':
    if len(argv) == 6:
        W = verbose_classifier(argv[4], int(argv[2]), int(argv[3]))
        #test_classifier(argv[2], W)
    else: 
        print('Please input the testfile and the training file')