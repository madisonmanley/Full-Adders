# Code for 3-bit Full Adder made by merging gates. It can only test the accuracy
# of addition or subtraction. It can also test the linear, clipped, and sigmoid 
# activation function. Also has the code to combine logical units. Noise can be
# added.


# Output format:
# x number of steps
# average accuracy % of test with x steps
# temperature value used in test


# Ex: 
# 100 // 100 steps
# 56.0 // accuracy %
# 0.5 // temperature value
#
# 1000
# 82.0625 // accuracy % of test with 1000 steps and t = 1
# 1
# etc...

import matplotlib.pyplot as plt
from math import tanh
from math import copysign
from math import sqrt
from random import uniform
from operator import concat
from operator import itemgetter
from functools import reduce 
import numpy as np
import pickle


# Function for sigmoid activation

def Imt(matrix, bias, vector, i):
    s = 0
    for m in range(len(vector)):
        s += matrix[i][m]*vector[m]
    #s += bias[i]
    
    return s
 
    
# Function for linear activation
    
def linear(num):
    
    # Based off the piece-wise function:
    # -1   x < -1
    # (1/a)x   -a<=x<=a
    # 1    x > a
    
    if num < -1:
        return -1
    elif num > 1:
        return 1
    else:
        return num
 
    
# Function for clipped activation
    
def clipped(num):
    
    # Based off the piece-wise function:
    # -1   x < -a
    # tanh(x)/-tanh(-a)  -a<=x<=a
    # 1    x > a
    
    if num < -3:
        return -1
    elif num > 3:
        return 1
    else:
        return tanh(num)/(-tanh(-3))

    
# Activation function based off circuit values
# Incomplete

def activation(num, voltage, activation):
    if num < -1:
        return -1
    elif num > 1:
        return 1
    else:
        b = min(voltage, key=lambda x:abs(x-num))
        i = voltage.index(b)
        if num > voltage[i+1]:
            return (activation[i]+activation[i+1])/2
        else:
            return (activation[i]+activation[i-1])/2


# Function that will combine two bits
            
def two_bits(matrix1, matrix2):

    # Dimension size of matrixes and seeing which nodes to connect
    # Bit counting starting with 1
    
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    dim = int(sqrt(np.size(matrix2))) - 1
    dim2 = int(sqrt(np.size(matrix1)))
    n1 = int(input("Matrix 1 - 1st bit: "))
    n2 = int(input("Matrix 1 - 2nd bit: "))
    n3 = int(input("Matrix 2 - 1st bit: "))
    n4 = int(input("Matrix 2 - 2nd bit: "))
    
    
    # Expanding the first matrix to new dimensions and adding element
    
    matrix1 = np.pad(matrix1, ((0,dim-1), (0, dim-1)), mode = 'constant', constant_values=0)
    matrix1[n1-1][n1-1] += matrix2[n3-1][n3-1]
    matrix1[n2-1][n2-1] += matrix2[n4-1][n4-1]
    

    # Deletes the Jii element and saves col & row
    
    row = np.delete(matrix2[n3-1:n3, :dim+1], n3-1)
    row2 = np.delete(matrix2[n4-1:n4, :dim+1], n4-1)
    col = np.delete(matrix2[:dim+1, n3-1:n3], n3-1)
    col2 = np.delete(matrix2[:dim+1, n4-1:n4], n4-1)
    

    # Adding weights
    
    matrix1[n1-1][n2-1] += matrix2[n3-1][n4-1]
    matrix1[n2-1][n1-1] += matrix2[n4-1][n3-1]
    
    
    # Deleting the added element
    
    row = np.delete(row, n4-2)
    col = np.delete(col, n4-2)
    row2 = np.delete(row2, n3-2)
    col2 = np.delete(col2, n3-2)
    
    
    # Deleting the columns 
    
    matrix2 = np.delete(np.delete(matrix2, (n3-1, n4-1), axis=0), (n3-1, n4-1), axis=1)


    # Filling in rows and columns
    
    for i in range(len(row)):
        matrix1[n1-1][dim2+i] = row[i]
        matrix1[n2-1][dim2+i] = row2[i]
        matrix1[dim2+i][n1-1] = col[i]
        matrix1[dim2+i][n2-1] = col2[i]

    for i in range(dim-1):
        for j in range(dim-1):
            matrix1[dim2+i][dim2+j] = matrix2[i][j]
        
    return matrix1 


# Function that will combine two p-bits
    
def two_combine(matrix1, matrix2):
    
    #turn to np.array
    
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

        
    # dimension size of second matrix and seeing which nodes to connect
    # bit count starting with 1
    
    dim = int(sqrt(np.size(matrix2))) - 1
    dim2 = int(sqrt(np.size(matrix1)))
    n1 = int(input("Matrix 1 bit: "))
    n2 = int(input("Matrix 2 bit: "))

    
    # expanding the first matrix to new dimensions and adding element
    
    matrix1 = np.pad(matrix1, ((0,dim), (0,dim)), mode = 'constant', constant_values=0)
    matrix1[n1-1][n1-1] += matrix2[n2-1][n2-1]
    
    
    # saving rows and columns into new vectors and deleting elements
    
    row = np.delete(matrix2[n2-1:n2, :dim+1], n2-1)
    col = np.delete(matrix2[:dim+1, n2-1:n2], n2-1)
    matrix2 = np.delete(np.delete(matrix2, (n2-1), axis=0), (n2-1), axis=1)


    # filling in rows and columns
    
    for i in range(dim):
        matrix1[n1-1][dim2+i] = row[i]
        matrix1[dim2+i][n1-1] = col[i]       
    
    
    # filling in the symmetric matrix
    
    for i in range(dim):
        for j in range(dim):
            matrix1[dim2+i][dim2+j] = matrix2[i][j]

    return matrix1 


# 3 bit vector
    
m = [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
     1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
     1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
     1, 1, 1, 1, -1, 1, 1, -1, -1]

# Collecting bits: A2, A1, A0, B2, B1, B0, Carry, S2, S1, S0

get = [70, 38, 5, 71, 39, 6, 96, 78, 46, 14]

avgerror = []
count = []
for i in range(1024):
    count.append(0)
    
    
# Loading 3-bit full adder matrix
    
with open('3FALU.p', 'rb') as f:
    J = pickle.load(f)


# Testing matrix for addition and subtraction 
    
test = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 17],
        [-1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 34],
        [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 51],
        [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 68],
        [-1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 85],
        [-1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 102],
        [-1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 119],
        [-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 129],
        [-1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 146],
        [-1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 163],
        [-1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 180],
        [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 197],
        [-1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 214],
        [-1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 231],
        [-1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 248],
        [-1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 258],
        [-1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 275],
        [-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 292],
        [-1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 309],
        [-1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 326],
        [-1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 343],
        [-1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 360],
        [-1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 377],
        [-1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 387],
        [-1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 404],
        [-1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 421],
        [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 438],
        [-1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 455],
        [-1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 472],
        [-1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 489],
        [-1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 506],
        [1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 516],
        [1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 533],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 550],
        [1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 567],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 584],
        [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 601],
        [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 618],
        [1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 635],
        [1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 645],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 662],
        [1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 679],
        [1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 696],
        [1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 713],
        [1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 730],
        [1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 747],
        [1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 764],
        [1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 774],
        [1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 791],
        [1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 808],
        [1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 825],
        [1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 842],
        [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 859],
        [1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 876],
        [1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 893],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 903],
        [1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 920],
        [1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 937],
        [1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 954],
        [1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 971],
        [1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 988],
        [1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1005],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1022]]


# bias value

h = 0
mvector = []


# number of steps and different temperature values

step_num = [100, 1000, 10000]
temp = [0.5, 1, 1.5, 2, 5]

# Testing all temperature values at all step values.

for s in range(len(step_num)):
    for t in range(len(temp)):
        for trial in range(64):
            # Keeping track of time-step
            for n in range(step_num[s]):
                # Gibbs Sampling: Updating each bit
                for bit in range(len(m)):
                    
                    # Clamped handle bits - bit 4, 12, 19, 25, 31, 13, 37, 45, 51,
                    # 57, 63, 69, 77, 83, 89, 95
                    
                    #get = [70, 38, 5, 71, 39, 6, 96, 78, 46, 14]
                    # For addition: bit 70, 38, 5, 71, 39, 6 are clamped to the test
                    # matrix values test[trial][0], test[trial][1], test[trial][2], 
                    # test[trial][3], test[trial][4], test[trial][5], respectfully. 
                    # The rest of the bits are commented out, manually.
                    
                    # For subtraction: bit 71, 39, 6, 96, 78, 46, 14 are clamped to the test
                    # matrix values test[trial][3], test[trial][4], test[trial][5], 
                    # test[trial][6], test[trial][7], test[trial][8], test[trial][9], 
                    # respectfully. The other bits are commented out, manually.
                    

                    if bit == 4:
                        m[bit] = 1
                    elif bit == 12:
                        m[bit] = 1
                    elif bit == 19:
                        m[bit] = 1
                    elif bit == 25:
                        m[bit] = 1
                    elif bit == 31:
                        m[bit] = -1
                    elif bit == 13:
                        m[bit] = -1
                    elif bit == 37:
                        m[bit] = 1
                    elif bit == 45:
                        m[bit] = 1
                    elif bit == 51:
                        m[bit] = 1
                    elif bit == 57:
                        m[bit] = 1
                    elif bit == 63:
                        m[bit] = -1
                    elif bit == 69:
                        m[bit] = 1
                    elif bit == 77:
                        m[bit] = 1
                    elif bit == 83:
                        m[bit] = 1
                    elif bit == 89:
                        m[bit] = 1
                    elif bit == 95:
                        m[bit] = -1
                    #get = [70, 38, 5, 71, 39, 6, 96, 78, 46, 14]
                    elif bit == 70:
                        m[bit] = test[trial][0]
                    elif bit == 38:
                        m[bit] = test[trial][1]
                    elif bit == 5:
                        m[bit] = test[trial][2]
                    elif bit == 71:
                        m[bit] = test[trial][3]
                    elif bit == 39:
                        m[bit] =test[trial][4]
                    elif bit == 6:
                        m[bit] = test[trial][5]
                    elif bit == 96:
                        m[bit] = test[trial][6]
                        """elif bit == 78:
                            m[bit] = test[trial][7]
                        elif bit == 46:
                            m[bit] = test[trial][8]
                        elif bit == 14:
                            m[bit] = test[trial][9]"""
                    else:
                        
                        mat = temp[t]*Imt(J, h, m, bit)
                        tan = tanh(mat)
                        #tan = tanh(mat+np.random.normal(0,5))
                        
                        # sigmoid functions are commented out to choose which function
                        # to test.
                        
                        
                        # regular sigmoid function
                        
                        m[bit] = (int(copysign(1, uniform(-1,1) + tan)))
                        
                        
                        # linear function
                        
                        #m[bit] = (int(copysign(1, uniform(-1,1) + linear(mat))))
                        
                        
                        # clipped sigmoid function
                        
                        #m[bit] = (int(copysign(1, uniform(-1,1) + clipped(mat))))
                        
            
                # Updating location
                
                u = [m[i] for i in get]
                u = [0 if x==-1 else x for x in u]
                mvector.append(reduce(concat, [str(item) for item in u]))  
            
            
            # finding the probabilities of each value
            
            for i in range(step_num[s]):
                count[int(mvector[i], 2)] += 1
             

            # Finding the accuracy of each test 
            
            error = (count[test[trial][10]]/step_num[s])*100
            avgerror.append(error)
           
            
            # Prints out the % accuracy of individual tests 
            #print (error,100-error)


            count = []
            for i in range(1024):
                count.append(0)
            del mvector[:]
            
            
       # Prints out the average accuracy of all tests combined
        
        print (step_num[s])
        print (np.mean(avgerror))
        print (temp[t])
        print ('\n')
        del avgerror[:]
 

