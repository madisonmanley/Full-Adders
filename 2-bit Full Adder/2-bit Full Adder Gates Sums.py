# Code for 2-bit Full Adder made by merging gates. It can only test the accuracy
# of reverse carry sum. It can also test the linear, clipped, and sigmoid 
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
     1, 1, 1, 1, -1, 1, 1, -1, -1]


# Collecting bits: A1, A0, B1, B0, Carry, S1, S0

get = [38, 5, 39, 6, 64, 46, 14]

avgerror = []
count = []
for i in range(128):
    count.append(0)


# Loading 2-bit full adder matrix

with open('2FALU.p', 'rb') as f:
    J = pickle.load(f)


# Testing matrix for addition and subtraction 

test = [[-1,-1,-1,0,-1,-1,-1],
        [1,-1,-1,60,84,108,-1],
        [-1,-1,1,9,33,-1,-1],
        [1,-1,1,93,117,-1,-1],
        [-1,1,-1,18,42,66,-1],
        [1,1,-1,126,-1,-1,-1],
        [-1,1,1,27,51,75,99]]


# bias value

h = 0
mvector = []


# number of steps and different temperature values

step_num = [100,1000,10000]
temp = [0.5, 1, 1.5, 2, 5]


# Testing all temperature values at all step values.

for s in range(len(step_num)):
    for t in range(len(temp)):
        for trial in range(7):
            # Keeping track of time-step
            for n in range(step_num[s]):
                # Gibbs Sampling: Updating each bit
                for bit in range(len(m)):
                    
                    mat = temp[t]*Imt(J, h, m, bit)
                    tan = tanh(mat)
                    
                    
                    # sigmoid functions are commented out to choose which function
                    # to test.
                    
                    
                    # regular sigmoid function
                    
                    m[bit] = (int(copysign(1, uniform(-1,1) + tan)))
                    
                    
                    # linear function
                    
                    #m[bit] = (int(copysign(1, uniform(-1,1) + linear(mat))))
                    
                    
                    # clipped sigmoid function
                    
                    #m[bit] = (int(copysign(1, uniform(-1,1) + clipped(mat))))
                    

                    # Clamped handle bits - bit 4, 12, 19, 25, 31, 13, 37, 45, 51, 57, 63
                    
                    # For reverse carry sum: bit 64, 46, 14 are clamped to the test
                    # matrix values test[trial][0], test[trial][1], test[trial][2], 
                    # respectfully. The other bits are commented out, manually.
                    

                    if bit == 4:
                        m[bit] = 1
                    if bit == 12:
                        m[bit] = 1
                    if bit == 19:
                        m[bit] = 1
                    if bit == 25:
                        m[bit] = 1
                    if bit == 31:
                        m[bit] = -1
                    if bit == 13:
                        m[bit] = -1
                    if bit == 37:
                        m[bit] = 1
                    if bit == 45:
                        m[bit] = 1
                    if bit == 51:
                        m[bit] = 1
                    if bit == 57:
                        m[bit] = 1
                    if bit == 63:
                        m[bit] = -1
                    if bit == 64:
                        m[bit] = test[trial][0]
                    if bit == 46:
                        m[bit] = test[trial][1]
                    if bit == 14:
                        m[bit] = test[trial][2]
                            
            
                # Updating location
                
                u = [m[i] for i in get]
                u = [0 if x==-1 else x for x in u]
                mvector.append(reduce(concat, [str(item) for item in u]))  
            
            
            # Finding the probabilities of each value
            
            for i in range(step_num[s]):
                count[int(mvector[i], 2)] += 1
                
                
            # Finding the accuracy of each test .
            
            error = 0
                    
            for i in range(4):
                if test[trial][i+3] == -1:
                    error = error
                else:
                    error += count[test[trial][i+3]]
            
            error = (error/step_num[s])*100
            avgerror.append(error)
                
            
            count = []
            for i in range(128):
                count.append(0)

            del mvector[:]
            
            
        # Prints out the average accuracy of all tests combined
        
        print (step_num[s])
        print (np.mean(avgerror))
        print (temp[t])
        print ('\n')
        del avgerror[:]