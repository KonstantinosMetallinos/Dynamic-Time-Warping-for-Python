# Dynamic-Time-Warping-for-Python
Dynamic Time Warping (DTW) function coded in Python. 

To use this file you will need to download and install Cython and Numpy.

To get the file up and running, you will need to download DTW_pyx.pyx and setup.py 

Save the files to desired location and open up a command line to said location and type: 
"python setup.py build_ext --inplace" 
This should generate some files and you will be ready to code! 

Once set up is ready, insert the following lines in your script: 

from DTW_pyx import * 
yourVariableName = NameDTW()

and you are ready to go. 

The function takes as an input 2 numpy arrays (of same dimentions) and returns their "Dynamic Time Warping dissimilarity score". An added option has been set to experiment with different powers but for the default result set it equal to 1. 


The code includes 4 types of DTW. 
getDistanceMatrix_Full: Vanlia version of DTW 
getDistanceMatrix_Percentage_Band: Ratanamahatana-Keogh Band with DTW 
getDistanceMatrix_Fixed_Band: Sakoe-Chiba Band with DTW 
getDistanceMatrix_Parallelogram_Band: Itakura Parallelogram version of DTW
getDistanceMatrix_Euc: The normal Euclidean Distance

The code has been fully Cythonised to get the smallest of computation times. 
