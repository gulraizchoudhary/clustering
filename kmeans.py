# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:40:58 2022

@author: G. I. Choudhary
"""

#Clustering Exercise 2
from data_utilities import data_utility as util
import numpy as np
import random as rand
import matplotlib.pyplot as plt

def euclidean_distance(x,y):   
    return np.sqrt(np.sum((x-y)**2))


def pairwise_distances(data):
    sum = 0.0
    for i in range(0,len(data)):
        for j in range(1,len(data)):
            distance= euclidean_distance(data[i],data[j])
            sum +=distance
        
    return sum

#/* this (example) objective function is sum of squared distances
#   of the data object to their cluster representatives */
def sse(data, partitions, centroids):
    sum = 0
    n=len(data)
    for i in range(n):
        sum = sum +(euclidean_distance(data[i],centroids[partitions[i]])**2)#Calculates total squared error (TSE)
    return sum/(n*len(data[0])) #calculates nMSE =(TSE/(N*V))
    


def getRandomPartition(data, k):
    return [rand.randint(1, k) for i in range(0,len(data))]



def getCentroids(data, partition, k):
    centroids = {}
    for i in set(partition):
        indexes = np.where(np.array(partition)==i)[0]
        sum =0.0
        for indx in indexes:
            sum+=data[indx]
        
        centroids[i] = sum/len(indexes)
               
    return centroids


def findNearest(data, centroids):
    partitions = []
    
    for d in data:
        distances ={}
        for key in centroids:
            distances[key] = euclidean_distance(d, centroids[key])
        
        min_d = min(distances.items(), key=lambda x: x[1])
        
        partitions.append(min_d[0])
        
    return partitions

def compareCentroids(prevC, newC):
    for key in prevC:
        if key in newC:
            comparison = prevC[key]== newC[key]
            if comparison.all() :
                return False
        else:
            return False
            
    return True

def getRandomCentroids(data, k):
    indices = set()
    while len(indices)<k:
        val = [rand.randint(1, len(data)) for i in range(0,len(data))]
        indices.add(val[0])
    centroids = {}
    for indx in indices:
        centroids[list(indices).index(indx)+1] = data[indx] 
                    
    return centroids

#Input: data, initial partitions, random centroids, k: number of clusters, it:number of iterations    
def kmeans(data, partition, centroids, k):
    sse_old = sse(data, partition, centroids)
    sse_new = 0.0
    while(sse_new!=sse_old):
        sse_old = sse_new
        centroids = getCentroids(data, partition, k)
        partition = findNearest(data, centroids)
        sse_new = sse(data, partition, centroids)
    return partition, centroids, sse_new


#import data from text file
data =  np.loadtxt("data/s1.txt")
k = 15

iterations = 10

#compute centroids based on random partitions
centroids = getRandomCentroids(data, k)
partition = findNearest(data, centroids)

se = []
for i in range(iterations):
    partition,centroids, sse2 = kmeans(data, partition, centroids, k)
    print(sse2)
    se.append(sse2)
    
#util.write_labels(partition, "partition.txt", mode = 'w')
#util.write_centroids(centroids.values(), "centroid.txt", mode = 'w')


#Converting data and centroid into matrix for plotting
cData = np.array(data)
keys = centroids.keys()

c = np.array([centroids[i] for i in keys])

#plotting the results
for i in np.unique(partition):
    plt.scatter(cData[partition == i , 0] , cData[partition == i , 1] , label = i)
plt.scatter(c[:,0] , c[:,1] , s = 80, color = 'k')
plt.show()