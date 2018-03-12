Chase Brown
SID 106015389
Machine Learning
Program 1: Linear Regression/Sum of Squares Error
Data Preprocessing

***Description**
Linear Regression program designed to implement the closed form equation on a given dataset.
Given a dataset it is broken down into two matrix. X being the 'attributes' of the set and Y
the 'solution'.  With this data the X and Y sets are broken into two more sets for each, Xtest and Ytest, Xtrain and Ytrain.
Given the Xtrain and Ytrain matrix the close form equation provides a solution or the 'weights'.  The weights matrix
is then used with the Xtest set to give a Y prediction.  This Y prediction is then used against the Y Test set
to determine the sum of squares error 'the difference between a predicted value and the actual true value'.  This
error provides a scale on how well the predicted value model was to the actual value based on the training.

Report includes 'My' closed form equation solution along with skLearns.LinearRegression() solution
numpy.linalg and statsmodels() for comparison of results.

Program:
Python 3.6
Written with Pycharm 2017 IDE on a Windows 10 OS PC.
requires:
numpy, matplotlib, pandas, statsmodels, and sklearn for libraries.
Running:
Simply run the data-preprocess.py.  Results will be printed out to myplots.pdf and report.doc

To run sklearn datasets see comments in data-preprocess.py