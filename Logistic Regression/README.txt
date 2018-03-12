Machine Learning
Program 2: Logistical Regression

***Description**
Using the closed form solution for gradient decent to calculate logistical regression.  Running the program
will use gradient decent with no data normalizing first against a 50/50 train test split doing kFold = 10
cross validation.  The data is then ran again for min-max, standardization, and regularization using
lambda[0, 1, 10, 100, 100] on the data that is standardized.
myplots.pdf is a collection of 4 graphs with the average for each collection of runs for comparison.
report.doc prints out each iteration of kfolds along with the average from the kfolds and the 50/50 split test.
Table.csv creates a csv with a collection of data for comparison with params used for the runs.

Program:
Python 3.6
Written with Pycharm 2017 IDE on a Windows 10 OS PC.
requires:
numpy, matplotlib, pandas, statsmodels, and sklearn for libraries.
Running:
Run the datapreprocess.py.  Results will be printed out to myplots.pdf (graphs) and report.doc and Table.csv
