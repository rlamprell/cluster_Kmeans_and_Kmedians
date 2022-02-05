############
Assignment 12
############

The file Assignment_2.py can be run to generate outputs for the questions in the assignment.

This will run the If __name__ == '__main__' function at the base of the file.  Which will in turn:
	-- Assign a numpy seed value for random (54 by default)
	-- Gather the data from the 4 files "veggies", "countries", "fruits" and "animals"
	-- Create a secondary data-set of normalised features
	-- Call the Questions() class and run for questions 3, 4, 5 and 6 from the assigment

A Sample output of this can be seen below.


NOTE: Ensure all the .data files are in the same location as Assignment_2.py when running




###############
Class Breakdown
###############

KMeans()
	-- intakes the number of clusters, k (default: 5) and epoches (default: 20)
	-- fit() can be called to run the K-Means optimisation process

KMedians()
	-- inherits from KMeans and has the same constructor
	-- fit() again can be called, but will this time run the K-Medians algorithm
		-- the only changes are the distance (manhattan) and and average (medians) calculations

GetData()
	-- import a single data file - assumes that the class label is at the start of the feature set

PreProcessing()
	-- concatenates the data extracts from each of the files
	-- can apply l2 normalisation for each object

Questions()
	-- template class to answer the questions from the assignment



##############################################################
Sample Output from running Assignment_2.py -- Graphs redacted:
##############################################################


==============================================
Question 3 -- B-CUBED for standard K-Means
----------------------------------------------
Running for... <class '__main__.KMeans'>

----------------------------------------------
clusters [1, 2, 3, 4, 5, 6, 7, 8, 9]
precisions [0.33 0.66 0.81 0.93 0.9  0.9  0.9  0.89 0.91]
recalls [1.   1.   0.98 0.93 0.82 0.59 0.51 0.48 0.5 ]
f_scores [0.47 0.75 0.87 0.93 0.85 0.7  0.62 0.59 0.63]
==============================================


==============================================
Question 4 -- B-CUBED for normalised K-Means
----------------------------------------------
Running for... <class '__main__.KMeans'>

----------------------------------------------
clusters [1, 2, 3, 4, 5, 6, 7, 8, 9]
precisions [0.33 0.66 0.82 0.79 0.93 0.93 0.89 0.94 0.93]
recalls [1.   1.   0.99 0.73 0.69 0.67 0.56 0.56 0.43]
f_scores [0.47 0.75 0.88 0.7  0.77 0.76 0.63 0.67 0.57]
==============================================


==============================================
Question 5 -- B-CUBED for standard K-Medians
----------------------------------------------
Running for... <class '__main__.KMedians'>

----------------------------------------------
clusters [1, 2, 3, 4, 5, 6, 7, 8, 9]
precisions [0.33 0.66 0.78 0.91 0.78 0.91 0.92 0.95 0.91]
recalls [1.   1.   0.92 0.91 0.61 0.6  0.51 0.59 0.43]
f_scores [0.47 0.75 0.82 0.91 0.61 0.68 0.63 0.71 0.55]
==============================================


==============================================
Question 6 -- B-CUBED for normalised K-Medians
----------------------------------------------
Running for... <class '__main__.KMedians'>

----------------------------------------------
clusters [1, 2, 3, 4, 5, 6, 7, 8, 9]
precisions [0.33 0.66 0.8  0.8  0.93 0.95 0.92 0.8  0.91]
recalls [1.   1.   0.97 0.74 0.75 0.82 0.56 0.51 0.51]
f_scores [0.47 0.75 0.86 0.7  0.81 0.86 0.64 0.5  0.64]
==============================================


