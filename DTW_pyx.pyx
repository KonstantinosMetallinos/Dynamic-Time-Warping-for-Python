#python setup.py build_ext --inplace

cimport cython
import numpy as np
cimport numpy as np

DTYPEint = np.int
DTYPEdouble = np.double
ctypedef np.int_t DTYPEint_t
ctypedef np.double_t DTYPEdouble_t


cdef inline double distance(double x, double y, int distance_power):
    #return abs(x-y)
    if(x-y<0):
        return(y-x)**distance_power
    else:
        return(x-y)**distance_power

cdef inline int int_max(int a, int b): return a if a>=b else b
cdef inline int int_min(int a, int b): return a if a<=b else b
cdef inline double double_min(double a, double b, double c): 
    if (a<=b)&(a<=c):
        return a 
    if (b<=a)&(b<=c):
        return b
    else:
        return c


class NameDTW():
    def __init__(self):
        pass

    #@cython.boundscheck(False)
    def getDistanceMatrix_Full(self, np.ndarray dataA, np.ndarray dataB, int distance_power):
            
        #from __main__ import dynamicTimeWarp_V_1 

        cdef int n_dataA
        cdef int n_dataB
        cdef int i
        cdef int j
        cdef DTYPEdouble_t dist
        cdef np.ndarray[DTYPEdouble_t,ndim=2] dtw_matrix      
        
        if np.all(dataA == dataB):
            n_dataA = dataA.shape[0]
            dtw_matrix = np.zeros([n_dataA,n_dataA],dtype=DTYPEdouble)
            for i in range(0, n_dataA-1):
                for j in range(i+1, n_dataA):
                    dist = self.dynamicTimeWarp_V_1(dataA[i,:], dataA[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
                    dtw_matrix[j,i] = dist                
        else:
            n_dataA, n_dataB = dataA.shape[0], dataB.shape[0]
            dtw_matrix = np.zeros([n_dataA, n_dataB],dtype=DTYPEdouble)
            for i in range(0, n_dataA):
                for j in range(0, n_dataB):
                    dist = self.dynamicTimeWarp_V_1(dataA[i,:], dataB[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
        return dtw_matrix


    #@cython.boundscheck(False)
    def getDistanceMatrix_Percentage_Band(self, np.ndarray dataA,np.ndarray dataB, int distance_power):
        
        #from __main__ import dynamicTimeWarp_V_2_0 

        cdef int n_data
        cdef int n_dataA
        cdef int n_dataB
        cdef int i
        cdef int j
        cdef DTYPEdouble_t dist
        cdef np.ndarray[DTYPEdouble_t,ndim=2] dtw_matrix      
        
        if np.all(dataA == dataB):
            n_data = dataA.shape[0]
            dtw_matrix = np.zeros([n_data,n_data],dtype=DTYPEdouble)
            for i in range(0, n_data-1):
                for j in range(i+1, n_data):
                    dist = self.dynamicTimeWarp_V_2_0(dataA[i,:], dataA[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
                    dtw_matrix[j,i] = dist                
        else:
            n_dataA, n_dataB = dataA.shape[0], dataB.shape[0]
            dtw_matrix = np.zeros([n_dataA, n_dataB],dtype=DTYPEdouble)
            for i in range(0, n_dataA):
                for j in range(0, n_dataB):
                    dist = self.dynamicTimeWarp_V_2_0(dataA[i,:], dataB[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
        return dtw_matrix

    #@cython.boundscheck(False)
    def getDistanceMatrix_Fixed_Band(self, np.ndarray dataA,np.ndarray dataB, int distance_power):
        
        #from __main__ import dynamicTimeWarp_V_2_1 

        cdef int n_data
        cdef int n_dataA
        cdef int n_dataB
        cdef int i
        cdef int j
        cdef DTYPEdouble_t dist
        cdef np.ndarray[DTYPEdouble_t,ndim=2] dtw_matrix      
        
        if np.all(dataA == dataB):
            n_data = dataA.shape[0]
            dtw_matrix = np.zeros([n_data,n_data],dtype=DTYPEdouble)
            for i in range(0, n_data-1):
                for j in range(i+1, n_data):
                    dist = self.dynamicTimeWarp_V_2_1(dataA[i,:], dataA[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
                    dtw_matrix[j,i] = dist                
        else:
            n_dataA, n_dataB = dataA.shape[0], dataB.shape[0]
            dtw_matrix = np.zeros([n_dataA, n_dataB],dtype=DTYPEdouble)
            for i in range(0, n_dataA):
                for j in range(0, n_dataB):
                    dist = self.dynamicTimeWarp_V_2_1(dataA[i,:], dataB[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
        return dtw_matrix

    #@cython.boundscheck(False)
    def getDistanceMatrix_Parallelogram_Band(self, np.ndarray dataA,np.ndarray dataB, int distance_power):
        
        #from __main__ import dynamicTimeWarp_V_2_2 

        cdef int n_data
        cdef int n_dataA
        cdef int n_dataB
        cdef int i
        cdef int j
        cdef DTYPEdouble_t dist
        cdef np.ndarray[DTYPEdouble_t,ndim=2] dtw_matrix      
        if np.all(dataA == dataB):
            n_data = dataA.shape[0]
            dtw_matrix = np.zeros([n_data,n_data],dtype=DTYPEdouble)
            for i in range(0, n_data-1):
                for j in range(i+1, n_data):
                    dist = self.dynamicTimeWarp_V_2_2(dataA[i,:], dataA[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
                    dtw_matrix[j,i] = dist                
        else:
            n_dataA, n_dataB = dataA.shape[0], dataB.shape[0]
            dtw_matrix = np.zeros([n_dataA, n_dataB],dtype=DTYPEdouble)
            for i in range(0, n_dataA):
                for j in range(0, n_dataB):
                    dist = self.dynamicTimeWarp_V_2_2(dataA[i,:], dataB[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
        return dtw_matrix

	#@cython.boundscheck(False)
    def getDistanceMatrix_Euc(self, np.ndarray dataA,np.ndarray dataB, int distance_power):
	    
        cdef int n_data
        cdef int n_dataA
        cdef int n_dataB
        cdef int i
        cdef int j
        cdef DTYPEdouble_t dist
        cdef np.ndarray[DTYPEdouble_t,ndim=2] dtw_matrix      
	    
        if np.all(dataA == dataB):
            n_data = dataA.shape[0]
            dtw_matrix = np.zeros([n_data,n_data],dtype=DTYPEdouble)
            for i in range(0, n_data-1):
                for j in range(i+1, n_data):
                    dist = self.dynamicTimeWarp_Euc(dataA[i,:], dataA[j,:], distance_power)    
                    dtw_matrix[i,j] = dist
                    dtw_matrix[j,i] = dist                
        else:
            n_dataA, n_dataB = dataA.shape[0], dataB.shape[0]
            dtw_matrix = np.zeros([n_dataA, n_dataB],dtype=DTYPEdouble)
            for i in range(0, n_dataA):
                for j in range(0, n_dataB):
                    dist = self.dynamicTimeWarp_Euc(dataA[i,:], dataB[j,:], distance_power)
                    dtw_matrix[i,j] = dist
        return dtw_matrix

    # Vanlia version Version_1.0
    #@cython.boundscheck(False)
    def dynamicTimeWarp_V_1(self, np.ndarray seqA,np.ndarray seqB, int distance_power):
        
        # create the cost matrix
        cdef int numRows = seqA.shape[0]
        cdef int numCols = seqB.shape[0]
        cdef int i
        cdef int j
        cdef np.ndarray[DTYPEdouble_t,ndim=2] cost = np.zeros([numRows, numCols],dtype=DTYPEdouble)
        
        # initialize the first row and column
        cost[0,0] = distance(seqA[0], seqB[0], distance_power) 
        for i in range(1, numRows):
            cost[i,0] = cost[i-1,0] + distance(seqA[i], seqB[0], distance_power)

        for j in range(1, numCols):
            cost[0,j] = cost[0,j-1] + distance(seqA[0], seqB[j], distance_power)

        # fill in the rest of the matrix
        for i in range(1, numRows):
            for j in range(1, numCols):
                cost[i,j] = double_min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1]) + distance(seqA[i], seqB[j], distance_power)

        return cost[-1,-1]



    # Ratanamahatana-Keogh Band
    #@cython.boundscheck(False)
    def dynamicTimeWarp_V_2_0(self, np.ndarray seqA,np.ndarray seqB, int distance_power):
        # create the cost matrix
        cdef int numRows = seqA.shape[0]
        cdef int numCols = seqB.shape[0]
        cdef int i
        cdef int j
        cdef np.ndarray[DTYPEdouble_t,ndim=2] cost = np.zeros([numRows, numCols],dtype=DTYPEdouble)+10**10
        cdef int R = int_max(numRows, numCols)*2//10 + 1 #math.ceil(max(numRows, numCols)*0.2)

        # initialize the first row and column
        cost[0,0] = distance(seqA[0], seqB[0], distance_power) 
        for i in range(1, R):   #needs to be up to R so anything else is unaccesible 
            cost[i,0] = cost[i-1,0] + distance(seqA[i], seqB[0], distance_power)

        for j in range(1, R):  #needs to be up to R so anything else is unaccesible 
            cost[0,j] = cost[0,j-1] + distance(seqA[0], seqB[j], distance_power)

        # fill in the rest of the matrix
        for i in range(1, numRows):
            for j in range(int_max(i-R,1), int_min(i+R,numCols)):
                cost[i,j] = double_min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1]) + distance(seqA[i], seqB[j], distance_power)

        return cost[-1,-1]



    # Sakoe-Chiba Band
    #@cython.boundscheck(False)
    def dynamicTimeWarp_V_2_1(self, np.ndarray seqA,np.ndarray seqB, int distance_power):
        # create the cost matrix
        cdef int numRows = seqA.shape[0]
        cdef int numCols = seqB.shape[0]
        cdef int i
        cdef int j
        cdef np.ndarray[DTYPEdouble_t,ndim=2] cost = np.zeros([numRows, numCols],dtype=DTYPEdouble)+10**10
        cdef int R = 5 #i<NumRows -5


        # initialize the first row and column
        cost[0,0] = distance(seqA[0], seqB[0], distance_power) 
        for i in range(1, R):   #needs to be up to R so anything else is unaccesible 
            cost[i,0] = cost[i-1,0] + distance(seqA[i], seqB[0], distance_power)

        for j in range(1, R):  #needs to be up to R so anything else is unaccesible 
            cost[0,j] = cost[0,j-1] + distance(seqA[0], seqB[j], distance_power)

        # fill in the rest of the matrix
        for i in range(1, (numRows)):
            if i<=(numRows-5): 
                R = 5
            else:
                R = numRows - i
            for j in range(int_max(i-R,1), int_min(i+R,numCols)):
                cost[i,j] = double_min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1]) + distance(seqA[i], seqB[j], distance_power)
                
        return cost[-1,-1]
           

    # Itakura Parallelogram
    #@cython.boundscheck(False)
    def dynamicTimeWarp_V_2_2(self, np.ndarray seqA,np.ndarray seqB, int distance_power):
        # create the cost matrix
        cdef int numRows = seqA.shape[0]
        cdef int numCols = seqB.shape[0]
        cdef int i
        cdef int j
        cdef np.ndarray[DTYPEdouble_t,ndim=2] cost = np.zeros([numRows, numCols],dtype=DTYPEdouble)+10**10
        cdef int R #= 5 #i<NumRows -5    

        # initialize the first row and column
        cost[0,0] = distance(seqA[0], seqB[0], distance_power) 
        for i in range(1, numRows):   
            cost[i,0] = cost[i-1,0] + distance(seqA[i], seqB[0], distance_power)

        for j in range(1, numCols): 
            cost[0,j] = cost[0,j-1] + distance(seqA[0], seqB[j], distance_power)

        # fill in the rest of the matrix
        for i in range(1, numRows):
            if i<=(numRows*3/8): 
                R = 2*i//3+1 #math.ceil(2/3*i)
            else:
                R = int_max(3*numRows//8 - i*2//5,1) #math.ceil(3*numRows/8 - i*2/5)
            for j in range(1, int_max(R,numCols)):
                cost[i,j] = double_min( cost[i-1,j], cost[i,j-1], cost[i-1,j-1]) + distance(seqA[i], seqB[j], distance_power)

        return cost[-1,-1]

    # Euclidean Distance
	#@cython.boundscheck(False)    
    def dynamicTimeWarp_Euc(self, np.ndarray seqA,np.ndarray seqB, int distance_power):
	    
	    # create the cost matrix
        cdef int j
        cdef int i = seqA.shape[0]
        cdef double cost = distance(seqA[0], seqB[0],distance_power) 
        for j in range(1, i):
                cost = cost + distance(seqA[j], seqB[j],distance_power)

        return cost


