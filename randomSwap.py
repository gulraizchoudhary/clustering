# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:40:58 2022

@author: G. I. Choudhary
"""

#Clustering Exercise 2
from data_utilities import data_utility as util
import numpy as np
import random as rand
import copy
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


def getSSE(data, partitions, centroids):
    sse_p = {}
    for i in set(partitions):
        indexes = np.where(np.array(partitions)==i)[0]
        se =0.0
        for indx in indexes:
            se+=(euclidean_distance(data[indx],centroids[i])**2)
        
        sse_p[i] = se/len(indexes)
               
    return sse_p



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

#removes one centroid randomly 
#and adds one centroid by randomly sampling a datapoint    
def swapCentroid(data, centroids):
    cIndx = rand.randint(1, len(centroids))
    dIndx = rand.randint(0, len(data)-1)
    centroids[cIndx] = data[dIndx]
    
    return centroids, cIndx

#Find nearest centroid
def nearestRepresentative(dPoint, centroids):
    k=len(centroids)
    j = 1
    for i in range(2,k+1):
        if euclidean_distance(dPoint,centroids[i]) < euclidean_distance(dPoint,centroids[j]):
           j = i
           
    return j

#
def localRepartition(partition,centroids,data,cIndx):
    
    #/* object rejection */
    for i in range(len(data)):
        if partition[i] ==cIndx:
            partition[i] = nearestRepresentative(data[i], centroids)
    #/* object attraction */
    for i in range(len(data)):
        if euclidean_distance(data[i],centroids[cIndx]) < euclidean_distance(data[i],centroids[partition[i]]):
            partition[i] = cIndx
    return partition

#Input: data, initial partitions, random centroids, k: number of clusters, it:number of iterations    
def kmeans(data, partition, centroids, k, it):
    
    for i in range(0,it):
        centroids = getCentroids(data, partition, k)
        partition = findNearest(data, centroids)
        #sse_new = sse(data, partition, centroids)
                        
    return partition, centroids


def randomSwap(data, k, iterations, kmean_it):
    #get random centroids
    centroids = getRandomCentroids(data, k)
    #compute partitions based on randomly choosen centroids
    partition = findNearest(data, centroids)
    
    #se = []
    sse_old = sse(data, partition, centroids)
    
    for i in range (0, iterations):
        #randomly choose a datapoint and swap it with random centroid
        kCentroids, cIndx = swapCentroid(data, copy.deepcopy(centroids))
        
        #kPartition = findNearest(data, kCentroids)
        kPartition = localRepartition(copy.deepcopy(partition), kCentroids, data, cIndx)
        kPartition, kCentroids = kmeans(data, kPartition, kCentroids, k, kmean_it)
        
        
        sse_new = sse(data, kPartition, kCentroids)
        
        if sse_new < sse_old:
            centroids = kCentroids
            partition = kPartition
            sse_old = sse_new        
            print("Iteration: "+str(i)+" MSE="+str(sse_new))
            #se.append(sse_new)
    return centroids, partition

#import data from text file
data =  np.loadtxt("data/s1.txt")
k = 15

iterations = 200
kmean_it = 2

    

centroids, partition =   randomSwap(data, k, iterations, kmean_it)
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