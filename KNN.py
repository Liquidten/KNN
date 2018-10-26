#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:30:03 2018

@author: sameepshah
"""
import csv
#import random
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# load irist dataset and converting them to floats

def Dataset(filename):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #print(dataset)
        del dataset[0]
        #print(dataset)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
        return dataset
        


# SIMILARITY CHECK FUNCTION 
# euclidean distance calcualtion


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)


# NEIGHBOURS - selecting subset with the smallest distance (Dev)

def getNeighbors(train2, dev, k):
	distances = []
	length = len(dev)-1
	for x in range(len(train2)):
		dist = euclideanDistance(dev, train2[x], length)
		distances.append((train2[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# NEIGHBOURS - (Test)
    
def getNeighbors2(train2, test, k):
	distances = []
	length = len(test)-1
	for x in range(len(train2)):
		dist = euclideanDistance(test, train2[x], length)
		distances.append((train2[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


# PREDICTED RESPONSE 

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


# MEASURING ACCURACY 


def Accuracy(dev, predictions):
	correct = 0
	for x in range(len(dev)):
		if dev[x][-1] in predictions[x]: 
			correct = correct + 1
			
	return (correct/float(len(dev))*100) 

def Accuracy2(test, predictions2):
	correct = 0
	for x in range(len(test)):
		if test[x][-1] in predictions2[x]: 
			correct = correct + 1
			
	return (correct/float(len(test))*100) 


# MAIN TO RUN KNN
if __name__=="__main__":
    
    
    PATH = "../HW_3/iris.csv"
    
    dataset = Dataset(PATH)
    train, test= train_test_split(dataset, test_size=0.15, random_state = 25)
    train2, dev = train_test_split(train, test_size = 0.15, random_state = 25)
    #print(test)
    print ('Train set: ' + str(len(train2)))
    print ('Test set: ' + str(len(test)))
    print('Dev set: ' + str(len(dev)))
    # generate predictions
    predictions=[]
    predictions2=[]
    k = 4
    for x in range(len(dev)):	
        neighbors = getNeighbors(train2, dev[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        
    accuracy = Accuracy(dev, predictions)
    print('Accuracy Dev Set: ' + str(accuracy) + '%')
    
    for x in range(len(test)):	
        neighbors2 = getNeighbors2(train2, test[x], k)
        result2 = getResponse(neighbors2)
        predictions2.append(result2)
       
    accuracy2 = Accuracy(test, predictions2)
    print('Accuracy Test Set: ' + str(accuracy2) + '%')
   
    
    
	

