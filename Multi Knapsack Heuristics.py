# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:38:56 2022
@author: Kamaldeep Kaur
@Id : 35984681
"""
import pandas as pd
import numpy as np

# Function to read and parse Input file
def ReadInput(filename):      
  dataFile = open(filename, "r")
  data = []
  headers = []
  rhss = []
  while True:
      rawdata = []  
      fline = dataFile.readline()
      fline = fline.strip()
      if len(fline) == 0: 
        break
      vars = fline.split(' ')
      header = [float(var) for var in vars]
      headers.append(header)
      while len(rawdata)!= (header[0])*(header[1]+1):    # Read all objective and constraint elements as speciifed in the first row of data set 
        theline = dataFile.readline()
        theline = theline.strip()
        if len(theline) == 0:
            break  
        readData = theline.split(" ")
        rawdata+= readData
      rawdata = [float(item) for item in rawdata]
      datum = np.array(rawdata)
      datum = np.reshape(datum,(int(header[1]+1),int(header[0])))  
      lastline =  dataFile.readline()
      lastline = lastline.strip()
      if len(lastline) == 0:
        break
      vars = lastline.split(' ')
      rhs = [float(var) for var in vars]    # Extract Rhs values for each dataset and append in rhss
      rhss.append(rhs)
    #Convert array to dataframe and transpose data to have Objective and Constraints as columns
      df = pd.DataFrame(datum)
      df = df.astype(float)
      df = df.transpose()
    # create column names in dataframe
      collist = []
      collen = len(df.columns)
      for i in range(collen):
        if i == 0:
            collist.append(str("Obj")) 
        else:
            collist.append(str("Con"+str(i)))

      df.columns = collist
      data.append(df)    #Append each dataset extracted from the file to list called "data"  
  dataFile.close()
  return data,headers,rhss



# Function to generate Constructive initial solution 
def InitialSolution(df,collen,rhs,elmts):
    x = np.random.randint(1, collen)  # Choose constraint from random generator to asses profit to Weight ration for knapsack 
    #print("Intial Solution column",x)
    df["P/W"] = df["Obj"]/(df[str("Con"+str(x))].replace(0,1))
    df = df.sort_values(by="P/W", ascending=False)  # Sort items as per Profit to Weight ratio in desc order
    sol = pd.DataFrame(columns = df.columns)
    conc , binsol = [0]*(collen-1) , [0]*elmts
    for i in range(len(df)):   # Validate rhs values constrains for each item entering the sack
         for j in range(collen-1):
             conc[j]= conc[j] + df[str("Con"+str(j+1))].values[i]
         if (any(conc[i] > rhs[i] for i in range(len(conc))) == True) :
             break 
         sol = sol.append(df[i:i+1])
    indexes = sol.index.values
    for i in indexes :
        binsol[i] = 1    # Initial solution in binary form
    return sol , binsol

# Function to calculate total weight for each constraint to check solution feasability
def TotProConstraints(df,binsol,collen):
   df['binsol'] = binsol
   conc = [0]*(collen-1)
   pro = 0
   pro = sum(df["Obj"]*df["binsol"])
   for j in range(collen-1):
       conc[j]= conc[j] + sum(df[str("Con"+str(j+1))]*df["binsol"])         
   return pro,conc

# Function to find neighbouring solution for Hill Climbing
def Negihboursol(binsol):
  n = len(binsol)
  result = np.copy(binsol)
  rnd = np.random.RandomState()  
  i = rnd.randint(n)   # pick an item at random and swap the values 0 to 1 / 1 to 0 
  if result[i] == 0:
    result[i] = 1
  elif result[i] == 1:
    result[i] = 0
  return result

# Hill Climbing method 
def HillClimbing(df,rhs,collen,binsol,numiter = 1000):
    # First evaluate the initial solution (its total proffit)
    sol = binsol
    df = df.set_index([df.index.values + 1])
    curr_pro = sum(df["Obj"]*sol)
    # Now carry out the main loop of the algorithm
    for i in range(numiter):
        temp = Negihboursol(sol)  # Find Neighbouring solution
        (neg_pro, negconlist) = TotProConstraints(df,temp,collen)
        if not(any(negconlist[i] > rhs[i] for i in range(len(negconlist))) == True) : # Check feasability of the neighbouring solution
           if (neg_pro > curr_pro):  # better solution, so accept adjacent
                sol = temp.copy(); 
                curr_pro = neg_pro
              
    df["binsol"] = sol
    finsol = df[df["binsol"]==1]   # Final solution of hill climbing
    return finsol


#  Simulated Annealing method 
def SimAnnealing(binsol,df,rhs,collen, starttemp= 100000,stoptemp = 0.001, alpha = 0.99):
    # solve using simulated annealing
  df = df.set_index([df.index.values + 1])  
  rnd = np.random.RandomState(5)
  currtemperature = starttemp  
  curr_binsol = binsol
  (curr_pro, conlist) = TotProConstraints(df,curr_binsol,collen)
  while currtemperature > stoptemp:   
    neg_binsol = Negihboursol(curr_binsol)
    (neg_pro, negconlist) = TotProConstraints(df,neg_binsol,collen)
    if not(any(negconlist[i] > rhs[i] for i in range(len(negconlist))) == True) :  # Validate feasability of the neighbouring solution 
        if (neg_pro >= curr_pro):  # better solution so accept adjacent
            curr_binsol = neg_binsol; curr_pro = neg_pro
        else:          # neighbouring solution is worst
            accept_p = np.exp( (neg_pro - curr_pro ) / currtemperature )  # if calculate accept_p value for difference in solution at curr temperation in the iteration
            p = rnd.random()
            if p < accept_p:  # if  randonm p is less then accept_p worse solution is picked anyway
                curr_binsol = neg_binsol; curr_pro = neg_pro
    currtemperature = currtemperature * alpha
  df["binsol"] = curr_binsol
  df = df[df["binsol"]==1]  # Final solution of simulated annearling
  return df 
         
##########################################################################################################
################################ Read and run the test files #############################################
##########################################################################################################

filename = input("Enter File Name :")
data, header,rhss = ReadInput(filename = filename)
heumethod = int(input("Enter 1 for Hill Climbing improvement or 2 for Simulated Annealing :"))
if heumethod == 1 :
     numiter = int(input("Enter number of iterations :"))
     HillClimbSol = []
     for k in range(len(data)):
         df = data[k]
         elements = len(df)
         rhs = rhss[k]
         collen = len(df.columns)
         for j in range(10):
             finalpro = 0
             for i in range(10):
                 intsol, binsol = InitialSolution(df,collen,rhs,elements)
                 intsol = intsol.set_index(intsol.index.values+1)
                 print("Intial Solution based on constructive Heuristics with Profit =" , 
                       sum(intsol["Obj"])," is \n",intsol["Obj"]) 
                 sol = HillClimbing(df, rhs,collen,binsol,numiter) 
                 if sum(sol["Obj"])  > finalpro:
                     finalpro = sum(sol["Obj"]) 
                     finalsol = sol
                     
             HillClimbSol.append([k,j,list(finalsol["binsol"].index.values),sum(finalsol["Obj"])])
     HillClimbSol = pd.DataFrame(HillClimbSol)
     HillClimbSol.columns = ["TestInstance","Iteration","Items","Solution"]
     HillClimbAllSol = HillClimbSol.groupby('TestInstance').agg({'Solution': ['mean', 'min', 'max']})
     HillClimbBestSol= HillClimbSol.loc[HillClimbSol.groupby(["TestInstance"])["Solution"].idxmax()]
     HillClimbAllSol.to_csv('HillClimb_Allresults.csv',float_format='%.2f')
     HillClimbBestSol.to_csv('HillClimb_BestSol.csv',float_format='%.2f',header=["Test Instance","Iteration","Items","Solution"])
     print("Maximum, Minimum, Avergae and Best solutions are saved in files HillClimb_Allresults.csv and HillClimb_BestSol.csv")
     print("Best Solution Obtained with following profit and items:","\n",HillClimbBestSol[["TestInstance","Solution","Items"]]) 
else :
    if  heumethod == 2: 
      stemp = int(input("Enter start temperature:")) #10000
      sptemp = float(input("Enter the end temperature :")) #1000
      alpha = float(input("Enter alpha:")) #0.98
      SimAneSol = []
      for k in range(len(data)):
         df = data[k]
         elements = len(df)
         rhs = rhss[k]
         collen = len(df.columns)
         for j in range(10):
                intsol, binsol = InitialSolution(df,collen,rhs,elements)
                intsol = intsol.set_index(intsol.index.values+1)
                print("Intial Solution based on constructive Heuristics with Profit =" , 
                      sum(intsol["Obj"])," is \n",intsol["Obj"]) 
                finalsol = SimAnnealing(binsol, df, rhs, collen, stemp, sptemp, alpha)
                SimAneSol.append([k,j,list(finalsol["binsol"].index.values),sum(finalsol["Obj"])])
      SimAneSol = pd.DataFrame(SimAneSol)
      SimAneSol.columns = ["TestInstance","Iteration","Items","Solution"]
      SimAneAllSol = SimAneSol.groupby('TestInstance').agg({'Solution': ['mean', 'min', 'max']})
      SimAneBestSol= SimAneSol.loc[SimAneSol.groupby(["TestInstance"])["Solution"].idxmax()]
      SimAneAllSol.to_csv('SimAnne_Allresults.csv',float_format='%.2f')
      SimAneBestSol.to_csv('SimAnne_BestSol.csv',float_format='%.2f',header=["Test Instance","Iteration","Items","Solution"])
      print("Maximum, Minimum, Avergae and Best solutions are saved in files SimAnne_Allresults.csv and SimAnne_BestSol.csv")
      print("Best Solution Obtained with following profit and items:","\n",SimAneBestSol[["TestInstance","Solution","Items"]]) 

    else :
         print("Option entered is invalid")
