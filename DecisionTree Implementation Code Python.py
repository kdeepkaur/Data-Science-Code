# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:08:47 2021

@author: Kamaldeep Kaur
@studentid: 35984681
"""

import numpy as np
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import time
import os
import sklearn . metrics as mt 
import tracemalloc
from sklearn.model_selection import KFold
from sklearn import datasets
import math
#import warnings
#warnings.filterwarnings("ignore")
########################################################################################################
#Read Files
#path = "D:\Modules\Programming Final Project\Final Report/"
#os.chdir(path)
#os.listdir()


# create tree class to store decision tree nodes data
class Treeclass:  
     def __init__(self,content,left = None,right=None):
        self.content = content
        self.left = left 
        self.right = right
        
# Function to sort and derive midpoint for each variable as per the feature number passed in parameter        
def findmidpoint(dataset,feature=0):
    mid = []
    dtlen = len(dataset)
    x = feature
    sortdata = sorted(dataset,key = lambda i:i[x])   # Sort the data based on feature values
    spos = 0
    for i in range(1,dtlen):   # Loop to identify the mid values of the input list features
         spos = spos+1      
         if (sortdata[i-1][x] + sortdata[i][x])/2 != sortdata[i-1][x]:
             midval = ((sortdata[i-1][x] + sortdata[i][x])/2) 
             mid.append([midval,spos,x])  
         else :
            continue
    mid = np.array(mid)
    return mid,sortdata
# Function to determine entropy for each iteration and find the maximum information gain for all the possible splits for different features
def Entropy(dataset,condprevious=''):
    dataset = np.array(dataset)
    target = dataset[0:,-1]
    tarlen = len(target)
    #left,right = [],[]  #Initiate variables 
    wdavgenpre = 2
    for j in range(len(dataset[0])-1):
        mid,sortdata = findmidpoint(dataset,j)
        sortdata = np.array(sortdata)
        for i in range(len(mid)):
            dt = sortdata[0:,-1]
            leftt = dt[0:int(mid[i][1])]
            leflen = len(leftt)
            rightt = dt[int(mid[i][1]):]
            righlen = len(rightt)
            tarcll = np.unique(leftt)
            tarclr = np.unique(rightt)
            pl = [0]*len(tarcll)
            pr = [0]*len(tarclr)
            for c in range(len(tarcll)):
                for j in range(leflen):    
                    if leftt[j] == tarcll[c]:
                        pl[c] = pl[c]+1     #Counters counting elements in left for different classes                
            for c in range(len(tarclr)):   
                for j in range(righlen): 
                    if rightt[j] == tarclr[c]:
                        pr[c] = pr[c]+1 
            pl =np.array(pl)
            pr = np.array(pr)
            lefent = sum((-1)*((pl/leflen)*np.log2(pl/leflen)))  # Calculate left and right entropy with all classes input in array pl and pr
            righent = sum((-1)*((pr/righlen)*np.log2(pr/righlen)))  
            wdavgent = ((leflen/tarlen)*lefent) + ((righlen/tarlen)*righent)   # Weighted average
            if wdavgenpre >= wdavgent:   # compare previous extropy to determine minimum entropy to find the max  infogain gain 
                   wdavgenpre = wdavgent
                   leftt,rightt = list(leftt), list(rightt)
                   lmax_class = max(set(leftt),key=leftt.count)
                   rmax_class = max(set(rightt),key=rightt.count)
                   condleft = "(row["+str(int(mid[i][2]))+"] <= "+str(mid[i][0])+")" #split for max info gain value left
                   condright = "(row["+str(int(mid[i][2]))+"] > "+str(mid[i][0])+")" #split for max info gain value right
                   splitvar = {'index':int(mid[i][2]),'value':mid[i][0],'postion':int(mid[i][1]),'conditionleft':condleft,'conditionright':condright,'lclass':lmax_class,'rclass':rmax_class}
    splitvar['conditionleft'] = condprevious +" & "+ splitvar['conditionleft'] if condprevious != '' else splitvar['conditionleft']    # Generate and store data with split condition, output classes for both left and right
    splitvar['conditionright'] = condprevious +" & "+ splitvar['conditionright'] if condprevious != '' else splitvar['conditionright']
    if wdavgenpre == 1 : return False
    #print(splitvar)
    return splitvar

#Ginnisplit function 
def Gini(dataset,condprevious=''):
    dataset = np.array(dataset)
    target = dataset[0:,-1]
    tarlen = len(target)
    # For split 1     
    #left,right = [],[]  #Initiate variables 
    Gini = 1
    for j in range(len(dataset[0])-1):
        mid,sortdata = findmidpoint(dataset,j)
        sortdata = np.array(sortdata)
        for i in range(len(mid)):
            dt = sortdata[0:,-1]  #Extract target variable from sorted data
            leftt = dt[0:int(mid[i][1])]
            leflen = len(leftt)
            rightt = dt[int(mid[i][1]):]
            righlen = len(rightt)
            tarcll = np.unique(leftt)
            tarclr = np.unique(rightt)
            pl = [0]*len(tarcll)
            pr = [0]*len(tarclr)
            for c in range(len(tarcll)):
                for j in range(leflen):    
                    if leftt[j] == tarcll[c]:
                        pl[c] = pl[c]+1     #Counters counting elements in left for different classes                
            for c in range(len(tarclr)):    
                for j in range(righlen): 
                    if rightt[j] == tarclr[c]:
                        pr[c] = pr[c]+1 
            probl = sum((np.array(pl)/len(leftt))**2)  if len(leftt)!= 0 else 0  # set p 0 if subset count of left or right is 0
            probr = sum((np.array(pr)/len(rightt))**2)  if len(rightt)!= 0 else 0  
            Ginitest = ((len(leftt)/tarlen) * (1-probl)) + ((len(rightt)/tarlen) * (1-probr))
            if Gini >= Ginitest :  # If Ginni value is less then the previous value it will store the new lower values
                   Gini = Ginitest
                   leftt,rightt = list(leftt), list(rightt)
                   lmax_class = max(set(leftt),key=leftt.count)
                   rmax_class = max(set(rightt),key=rightt.count)
                   condleft = "(row["+str(int(mid[i][2]))+"] <= "+str(mid[i][0])+")" #split for minimum gini value left
                   condright = "(row["+str(int(mid[i][2]))+"] > "+str(mid[i][0])+")" #split for minimum gini value right
                   splitvar = {'index':int(mid[i][2]),'value':mid[i][0],'postion':int(mid[i][1]),'conditionleft':condleft,'conditionright':condright,'lclass':lmax_class,'rclass':rmax_class}
    splitvar['conditionleft'] = condprevious +" & "+ splitvar['conditionleft'] if condprevious != '' else splitvar['conditionleft']
    splitvar['conditionright'] = condprevious +" & "+ splitvar['conditionright'] if condprevious != '' else splitvar['conditionright']  # genrate and store data for split , like split condition of left right and classes in left or right
    if Gini == 1 : return False
    return splitvar

#Function to split data at root node
def firstsplit(dataset,splitm):
    if splitm == 'Gini':
       return  Treeclass(Gini(dataset))
    elif splitm == 'Entropy':
       return  Treeclass(Entropy(dataset))
    else:
        return "Incorrect split method passed"
    
# Recursive function to spliting data at all levels except at root node.
def recursionleftright(Tree,dataset,splitm,maxdepth=-1,min_samp_split=2,depth=0):
    depth = depth+1
    sp = Tree.content
    x = sp['index']
    minnum = min_samp_split if isinstance(min_samp_split, int) else math.ceil((min_samp_split*len(dataset)))
    sortdata = sorted(dataset,key = lambda i:i[x])
    leftdataset = np.array(sortdata[0:sp['postion']])
    rightdataset = np.array(sortdata[sp['postion']:])
    leftdtlen = len(np.unique(leftdataset[:,-1]))
    rightdtlen = len(np.unique(rightdataset[:,-1]))
    leftlen = len(leftdataset)
    rightlen = len(rightdataset)
    if depth == maxdepth:   # If depth reachs the max depth then return 
       return
  # return when both the subset (left and right are pure and no further split required)
    if leftdtlen ==1 and rightdtlen ==1 and leftlen < minnum and rightlen < minnum :  # return if both left and right nodes dont have multiple classes or minimum element not found
        return 
    #Left 
    if leftdtlen != 1 and leftlen >= minnum:    #if minimum num of elements are not found for split or only one class found, return 
        if splitm == 'Gini':
            Tree.left = Treeclass(Gini(leftdataset,sp['conditionleft']))
        elif splitm == 'Entropy':
            Tree.left = Treeclass(Entropy(leftdataset,sp['conditionleft']))
        recursionleftright(Tree.left,leftdataset,splitm,maxdepth,minnum,depth)   # Call the recursive function again for next iteration 
    #Right
    if rightdtlen !=1 and rightlen >= minnum:
        if splitm == 'Gini':
            Tree.right = Treeclass(Gini(rightdataset,sp['conditionright']))
            
        elif splitm == 'Entropy':
            Tree.right = Treeclass(Entropy(rightdataset,sp['conditionright']))
        recursionleftright(Tree.right,rightdataset,splitm,maxdepth,minnum,depth)

##Code  for traversing through Tree cited from https://www.analyticsvidhya.com/blog/2021/11/traverse-trees-using-level-order-traversal-in-python/
def printtree(Tree):
    test= []
    Treelist = []
    Treelist.append(Tree)            # append treelist with content of the tree after every pop 
    while Treelist != []:    # make l
       x = Treelist.pop(0)
       if x.left == None  and x.right == None:
           test.append(('both',x.content))
       elif x.left == None:
           test.append(('left',x.content))
       elif x.right == None:
           test.append(('right',x.content))
       if x.left:
          Treelist.append(x.left)
       if x.right:
          Treelist.append(x.right)
    return test
#
           
##Predict classes for test data
def predict(Tree,dataset):
    Treelist = []
    condneeded = []
    y_pre= []
    Treelist.append(Tree)
    while Treelist != []:     # It extract the condition from the Tree nodes and store it in condneeded list
       x = Treelist.pop(0)
       if x.left == None and x.right == None:
          condneeded.append([x.content['conditionleft'],x.content['conditionright'],x.content['lclass'],x.content['rclass']])
       elif x.left == None:
          condneeded.append([x.content['conditionleft'],'',x.content['lclass'],''])
          Treelist.append(x.right)
       elif x.right == None:
          condneeded.append(['',x.content['conditionright'],'',x.content['rclass']])
          Treelist.append(x.left)
       else :
          Treelist.append(x.left) 
          Treelist.append(x.right)
    for row in dataset:    #Use condneeded to filter data and assign classes stored in the same variable (which are true for that condition)
        validation = []
        for i in range(len(condneeded)):
            if condneeded[i][0] != '':
                if eval(condneeded[i][0]):
                    y_pre.append((condneeded[i][2])) #if eval(condneeded[i][0]) else None
                    validation.append(eval(condneeded[i][0]))
            if condneeded[i][1] != '':
                if eval(condneeded[i][1]):
                   y_pre.append((condneeded[i][3]))  
                   validation.append(eval(condneeded[i][1]))
                
    return y_pre      


    
################################# Testting and Analysis ###############################################
#Testing

############################### Functions to train tree and predict ######################################################
#Implemented tree processing function, Used to run decision tree for multiple combindations and stores the input parameters and output like classification metrics, time taken and memory used.
# It prepares the data for intermediate file
def RunImplDT(dt,x_test,y_test,crit,dep,minsamp=0):
    before_time = time.process_time_ns()  # Time in nano seconds
    tracemalloc.start()
    mod = firstsplit(dt,crit)
    recursionleftright(mod,dt,crit,dep,minsamp)
    peakmem = tracemalloc.get_traced_memory()[1]/1000000 #peak memory usage in MB
    tracemalloc.stop()
    after_time = time.process_time_ns()
    #Time taken to fit the model in microseconds
    processingTime = (after_time - before_time)/(10**3)
    predicted_y = predict(mod,x_test)
    #Metrics output 
    DTmodelout= pd.DataFrame(mt.classification_report(y_test,predicted_y,output_dict = True)).transpose()
    DTmodeldata = DTmodelout.loc[set(DTmodelout.index) - set(['accuracy'])]
    Noofrows = (DTmodeldata.shape[0])
    DTmodeldata['Accuracy'] = list(np.unique(DTmodelout.loc['accuracy']))*Noofrows
    DTmodeldata['TrainingSize'] = [len(x_train)]*Noofrows
    DTmodeldata['TrainTime(MicroSec)']  = [processingTime]*Noofrows
    DTmodeldata['Criterion'] = [crit]*Noofrows
    DTmodeldata['TreeDepth'] = [dep]*Noofrows 
    DTmodeldata['Memory(MB)']= [peakmem]*Noofrows
    DTmodeldata['Minsplit']= [math.ceil(minsamp*len(x_train))]*Noofrows if minsamp !=2 else [2]*Noofrows
    return DTmodeldata

#SKlearn tree processing function , used to run classifier for multiple combindations and stores the input parameters and output like classification metrics, time taken and memory used.
# It prepares the data for intermediate file
def RunSkDT(dt,x_test,y_test,crit,dep=None,minsamp=0):
    before_time = time.process_time_ns()
    tracemalloc.start()
    if dep == 0:
        clf = DecisionTreeClassifier(criterion= crit,random_state=0,min_samples_split=minsamp)
    else :
        clf = DecisionTreeClassifier(criterion= crit,random_state=0,min_samples_split=minsamp,max_depth= dep)
    clf.fit(x_train,y_train)
    peakmem = tracemalloc.get_traced_memory()[1]/1000000 #peak memory usage in MB
    tracemalloc.stop()
    after_time = time.process_time_ns()
    processingTime = (after_time - before_time)/(10**3)  # time in microsecond
    predicted_y = clf.predict(x_test)       
    Skmodelout= pd.DataFrame(mt.classification_report(y_test,predicted_y,output_dict = True)).transpose()
    Skmodeldata = Skmodelout.loc[set(Skmodelout.index) - set(['accuracy'])]
    Noofrows = (Skmodeldata.shape[0])
    Skmodeldata['Accuracy'] = list(np.unique(Skmodelout.loc['accuracy']))*Noofrows
    Skmodeldata['TrainingSize'] = [len(x_train)]*Noofrows
    Skmodeldata['TrainTime(MicroSec)']  = [processingTime]*Noofrows
    Skmodeldata['Criterion'] = [crit]*Noofrows
    Skmodeldata['TreeDepth'] = [dep]*Noofrows 
    Skmodeldata['Memory(MB)']= [peakmem]*Noofrows
    Skmodeldata['Minsplit']= [math.ceil(minsamp*len(x_train))]*Noofrows if minsamp !=2 else [2]*Noofrows
    return Skmodeldata       

############################## Testing with Mock Data #################################

x = np.array([[1,8],[2,1],[3.5,1],[7,2],[8,0],[11,9],[26,3]])
y = np.array([[0],[1],[2],[2],[1],[1],[0]])
x, y = datasets.load_iris(return_X_y=True)
y = np.reshape(y, (-1, 1))
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.3)
dt = np.concatenate((x_train,y_train),axis=1)

mod = firstsplit(dt,"Gini")
recursionleftright(mod,dt,"Gini")

predicted_y = predict(mod,x_test)
print(mt.classification_report(y_test, predicted_y))


clf = DecisionTreeClassifier(criterion= "gini",random_state=0)
clf.fit(x_train,y_train)
predicted_y = clf.predict(x_test)       
print(mt.classification_report(y_test, predicted_y))



############################## Load Iris Dataset ######################################
x, y = datasets.load_iris(return_X_y=True)

#Check corelation in input variable and passes the independent variable
print(pd.DataFrame(x).corr())
# Since iris dataset has high corealtion between Sepal Length and Sepal Width, will remove Sepal width for classification purposes
x = x[:,:3]
y = np.reshape(y, (-1, 1))  # Reshape Iris target variable to 2 dimensional array

############################## 5 Fold cross Validation datasets########################

#Prepare 5 fold cross validation data for Iris dataset
kf = KFold(n_splits=5,random_state =32,shuffle = True)

################### Generate Iris Output file for implemented tree #####################
# Implemented tree Classifier

finaldataset = pd.DataFrame()  
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = np.concatenate((x_train,y_train),axis=1)
    for crit in ['Entropy','Gini']:
        for dep in range(0, 5, 1):
            #if dep == 2:
                #continue
            if dep == 0:
                for minsamp in range(0,20,2):
                    minsamp = minsamp/100  if minsamp !=0 else 2
                    DTmodeldata = RunImplDT(dt,x_test,y_test,crit,-1,minsamp)
                    finaldataset= finaldataset.append(DTmodeldata)
            else:
                    DTmodeldata = RunImplDT(dt,x_test,y_test,crit,dep,minsamp=2)
                    finaldataset= finaldataset.append(DTmodeldata)
finaldataset = finaldataset.rename(columns= {"precision" : "Precision","recall":"Recall","f1-score":"F1Score","support":"Support"})  
finaldataset = finaldataset.round({"Precision":2,"Recall":2,"F1Score":2,"Accuracy":2,"TrainTime(MicroSec)":3,"Memory(MB)":3})          
finaldataset["TreeDepth"].replace({-1:max(finaldataset["TreeDepth"])+1}, inplace=True) 
finaldataset.to_csv("DToutput_irisdata.csv")           

################### Generate Iris Output file for SKlearn tree #####################
# Sklearn tree Classifier

skfinaldataset = pd.DataFrame() 
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dt = np.concatenate((x_train,y_train),axis=1)
    for crit in ['entropy','gini']:
        for dep in range(0, 5, 1):
            before_time = time.process_time_ns()
            tracemalloc.start()
            if dep == 0:
               for minsamp in range(0,20,2):
                    minsamp = minsamp/100  if minsamp !=0 else 2
                    Skmodeldata = RunSkDT(dt,x_test,y_test,crit,0,minsamp)
                    skfinaldataset= skfinaldataset.append(Skmodeldata)
            else:
                    Skmodeldata = RunSkDT(dt,x_test,y_test,crit,dep,minsamp=2)
                    skfinaldataset= skfinaldataset.append(Skmodeldata)
skfinaldataset["TreeDepth"].replace({0:max(skfinaldataset["TreeDepth"])+1},inplace=True)            
skfinaldataset = skfinaldataset.rename(columns= {"precision" : "Precision","recall":"Recall","f1-score":"F1Score","support":"Support"})  
skfinaldataset = skfinaldataset.round({"Precision":2,"Recall":2,"F1Score":2,"Accuracy":2,"TrainTime(MicroSec)":3,"Memory(MB)":3})          
skfinaldataset.to_csv("Skoutput_irisdata.csv")           

#############################################################################################################################################################################
###Following sectio generates output for banknote data and take long time, hence commented, Can be uncommented to test if needed ############################################
################################# BankNote Dataset ##########################################################
#Run and prepare output files for computational performaces with different training data sizes of banknote datset
######### This takes 10-20 mins to generate file due to multi parameters and file sizes ######################
# =============================================================================
# 
# x = pd.read_csv("data_banknote_authentication_input.csv").to_numpy()
# y = pd.read_csv("data_banknote_authentication_output.csv").to_numpy()
# 
# ###################### Generate banknote performance file for implemented tree ####################################
# #Implementd tree classifier
# finaldiffsizes = pd.DataFrame()   
# for i in range(7,1,-1):
#    split = i/10
#    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=split,random_state = 32)
#    dt = np.concatenate((x_train,y_train),axis=1)
#    for crit in ['Entropy','Gini']:
#         for dep in range(0, 5, 1):
#             if dep == 0:
#                for minsamp in range(0,10,1):
#                     minsamp = minsamp/100  if minsamp !=0 else 2
#                     DTmodeldata = RunImplDT(dt,x_test,y_test,crit,-1,minsamp)
#                     finaldiffsizes= finaldiffsizes.append(DTmodeldata)
# 
#             else:
#                     DTmodeldata = RunImplDT(dt,x_test,y_test,crit,dep,2)
#                     finaldiffsizes= finaldiffsizes.append(DTmodeldata)
# finaldiffsizes = finaldiffsizes.rename(columns= {"precision" : "Precision","recall":"Recall","f1-score":"F1Score","support":"Support"})  
# finaldiffsizes = finaldiffsizes.round({"Precision":2,"Recall":2,"F1Score":2,"Accuracy":2,"TrainTime(MicroSec)":3,"Memory(MB)":3})          
# finaldiffsizes["TreeDepth"].replace({-1:max(finaldiffsizes["TreeDepth"])+1}, inplace=True) 
# finaldiffsizes.to_csv("D:\Modules\Programming Final Project\Final Report/DToutput_banknote.csv")           
# 
# ##################### Generate banknote performance file for Sklearn tree ###########################################
# #Sklearn tree classifier
# skfinaldiffsizes = pd.DataFrame() 
# for i in range(9,1,-1):
#    split = i/10
#    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=split,random_state = 32)
#    dt = np.concatenate((x_train,y_train),axis=1)
#    for crit in ['entropy','gini']:
#         for dep in range(0, 5, 1):
#             before_time = time.process_time_ns()
#             tracemalloc.start()
#             if dep == 0:
#                for minsamp in range(0,10,1):
#                     minsamp = minsamp/100  if minsamp !=0 else 2
#                     Skmodeldata = RunSkDT(dt,x_test,y_test,crit,0,minsamp)
#                     skfinaldiffsizes= skfinaldiffsizes.append(Skmodeldata)
#             else:
#                     Skmodeldata = RunSkDT(dt,x_test,y_test,crit,dep,minsamp=2)
#                     skfinaldiffsizes= skfinaldiffsizes.append(Skmodeldata)
# 
# skfinaldiffsizes["TreeDepth"].replace({0:max(skfinaldiffsizes["TreeDepth"])+1},inplace=True)            
# skfinaldiffsizes = skfinaldiffsizes.rename(columns= {"precision" : "Precision","recall":"Recall","f1-score":"F1Score","support":"Support"})  
# skfinaldiffsizes = skfinaldiffsizes.round({"Precision":2,"Recall":2,"F1Score":2,"Accuracy":2,"TrainTime(MicroSec)":3,"Memory(MB)":3})          
# skfinaldiffsizes.to_csv("D:\Modules\Programming Final Project\Final Report/skoutput_banknote.csv")           
# 
# 
# =============================================================================
