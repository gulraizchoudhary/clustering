# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:59:46 2022

@author: gulrch
"""
import numpy as np

def write_labels(data_seq, file_path, mode = 'w'):
    with open(file_path, mode) as f:
        j = 0
        for i in data_seq:
            f.write(str(i))    
            f.write('\n')
            j = j + 1
def write_centroids(centroids, file_path, mode = 'w'):
    with open(file_path, mode) as f:
        j = 0
        for c in centroids:
            strippedText = str(c).replace('[','').replace(']','').replace('\'','').replace('\"','')
            f.write(strippedText)    
            f.write('\n')
            j = j + 1
            