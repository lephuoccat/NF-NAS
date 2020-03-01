#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:39:30 2020

@author: catle
"""

# importing library
import csv
from itertools import islice

# Sliding window
def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# load dataset
def LoadData(filename):
    # initializing the titles and rows list 
    fields = [] 
    rows = []
    
    # load csv file 
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
        
        # extracting field names through first row 
        fields = next(csvreader)
        
        # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 
            
        # get total number of rows 
        print("Total number of rows: %d"%(csvreader.line_num)) 
    
    # printing the field names 
    print('Field names: ' + ', '.join(field for field in fields)) 
    return row