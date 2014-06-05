'''
Created on Jun 2, 2014

@author: Sean
'''

from math import sqrt
from sys import argv
import random

def add_vectors(a, b):
    '''Add vectors a and b '''
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

def print_bootstrap(sets):
    '''Print bootstrap sample sets'''
    for i in range(len(sets)):
        print("Boostrap sample set",end=' ')
        print(i+1,end='')
        print(':')
        for j in range(len(sets[i])):
            for k in range(len(sets[i][j])-1):
                print(sets[i][j][k],end=' ')
            print('-',end=' ')
            if sets[i][j][-1] == 1:
                print("True")
            else:
                print("False")
        print()

def train_classifier(filename, T, size):
    '''Read a training file and learn the parameters for each class'''
    training_file = open(filename)
    
    D, nTrue, nFalse = map(int, training_file.readline().split())
    input = []
    
    count = 0 # Line and class counters
    '''Go through training data and append to list while applying 1 for true and 0 for false'''
    for line in training_file:       
        x = list(map(float, line.split()))
        if count < nTrue:
            x.append(1)
        else:
            x.append(0)
        input.append(x)
        count+=1
    training_file.close()
    
    '''Get bootstrap sets'''
    total_set = []
    for i in range(T):
        bootstrap_set = []
        for j in range(size):
            set = random.randint(0, nTrue+nFalse-1)
            temp = input[set]
            bootstrap_set.append(temp)
        total_set.append(bootstrap_set)
    
    '''Get centroids then get w and t'''    
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
    
    '''Package together W vector and bootstrap sets for printing later'''    
    package = []
    package.append(W)
    package.append(total_set)    
    return package

def verbose_test_classifier(filename, input):
    testing_file = open(filename)
    
    W = input[0]
    bootstrap = input[-1]
    class_list = []
    D, nTrue, nFalse = map(int, testing_file.readline().split())
    tp, fp, tn, fn = 0,0,0,0
    count = 0
    '''Go line by line and test'''
    for line in testing_file:
        x = list(map(float, line.split()))
        '''Add -1 to end of vector'''
        x.append(-1)
        vote = 0 # Vote for the class >=0 is True
        for i in range(len(W)):
            if dot_product(W[i], x) >= 0:
                vote += 1
            else:
                vote += -1
        
        if count < nTrue:
            actual_class = 0
        else:
            actual_class = 1
        # 0 = true, 1 = false; 0 = correct, 1 = false positive, 2 = false negative
        if actual_class == 0:
            if vote >= 0:
                tp+=1
                x[-1] = 0 # Mark true or false
                x.append(0) # mark correct, fp, or fn
            else:
                fp+=1
                x[-1] = 1
                x.append(1)
        else:
            if vote >= 0:
                fn+=1
                x[-1] = 0
                x.append(2)
            else:
                tn+=1
                x[-1] = 1
                x.append(0)
        class_list.append(x)
        count+=1
        
    testing_file.close()
    
    '''Print out non-verbose stuff'''
    print("Positive examples:",end=' ')
    print(tp+fp)
    print("Negative examples:",end=' ')
    print(tn+fn)
    print("False positives:",end=' ')
    print(fp)
    print("False negatives:",end=' ')
    print(fn)
    print()
    
    '''Print bootstrap sets'''
    print_bootstrap(bootstrap)
    
    '''Print classification'''
    print("Classification:")
    for i in range(len(class_list)):
        for j in range(len(class_list[i])-2):
            print(class_list[i][j],end=' ')
        print("-",end=' ')
        if class_list[i][-2] == 0:
            print("True (",end='')
        else:
            print("False (",end='')
        if class_list[i][-1] == 0:
            print("correct)")
        elif class_list[i][-1] == 1:
            print("false positive)")
        else:
            print("false negative)")
            
def normal_test_classifier(filename, input):
    testing_file = open(filename)
    
    W = input[0]
    class_list = []
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
        # 0 = true, 1 = false; 0 = correct, 1 = false positive, 2 = false negative
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
        count+=1
        
    testing_file.close()
    
    print("Positive examples:",end=' ')
    print(tp+fp)
    print("Negative examples:",end=' ')
    print(tn+fn)
    print("False positives:",end=' ')
    print(fp)
    print("False negatives:",end=' ')
    print(fn)

if __name__ == '__main__':
    if len(argv) == 6:
        W = train_classifier(argv[4], int(argv[2]), int(argv[3]))
        verbose_test_classifier(argv[5], W)
    elif len(argv) == 5:
        W = train_classifier(argv[3], int(argv[1]), int(argv[2]))
        normal_test_classifier(argv[4], W)
    else: 
        print('Please input the testfile and the training file')