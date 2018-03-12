# Chase Brown
# SID 106015389
# Machine Learning
# Program 1: Linear Regression/Sum of Squares Error

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets

# import datasets
dataset = pd.read_csv('Features_Variant_1.csv', header=None)
dataset.drop([37], axis=1, inplace=True)
print(dataset.shape)
samples = dataset.shape[0]
attr = dataset.shape[1]-1
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
# **********************************************************************************************
# To run a sklearn dataset comment out above 6 lines of code and uncomment below 3 lines of code
# **********************************************************************************************
#X, Y = datasets.load_boston(return_X_y=True)
#samples = X.shape[0]
#attr = X.shape[1]-1


# StandarScaler Standardized features by removing the mean and scaling to unit variance.
sc = StandardScaler()

def lReg(x, y):
    # add bias to x_train
    x = np.insert(x, 0, 1, axis=1)
    #print(x.shape)
    weights = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    return weights


def lPred(weights, x_test):
    # add bias to x_test
    x_test = np.insert(x_test, 0, 1, axis=1)
    Ypred = np.dot(x_test, weights)
    return Ypred


def plotG(xaxis, yaxis, h, title):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('Ypred')
    plt.ylabel('Y Actual')
    plt.scatter(xaxis, yaxis, color='k', alpha=0.5)
    plt.plot(h)
    plt.show()
    return fig


def plotS(sums, title):
    fig = plt.figure()
    plt.title(title)
    plt.plot(sums, '-o', color='k', alpha=0.7)
    plt.xlabel('Set')
    plt.ylabel('SSE Value')
    x = np.array([0, 1, 2])
    my_xticks = ['A', 'B', 'C']
    plt.xticks(x, my_xticks)
    plt.xticks()
    plt.show()
    return fig


f = open('report.doc', 'w')
pp = PdfPages('myplots.pdf')
f.write('Chase Brown\nSID: 106015389\nMachine Learning: Program 1 Linear Regression\n\n\n')
f.write('1. How many samples are there in the dataset?\n')
s = str(samples)
f.write(s+' samples\n')
f.write('How many attributes per sample?\n')
a = str(attr)
f.write(a+' attributes if we exclude the column we had to drop and the Y attribute.\n\n\n')

# Prepare training and test set for A 80%:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123456)
# w for linear hypothesis
#print(x_train.shape)
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)
f.write('Sum of Squared Error 0.8 training\n')
# Sum of Squares Error
sse_a = 0.5*sum((y_test - Ypred)**2)
# change range 50 for boston 1000 for features

# plot1 = plotG(abs(Ypred), y_test, h, '0.8 Training')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_a_lr = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np = 0.5*sum((numYpred-y_test)**2)
result = pd.DataFrame({"My": sse_a, "SkLearn": sse_a_lr, "numpy": sse_np, "Statsmodels": sse_sm}, index=[0])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

# Prepare training and test set for B 50%:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=123456)
# w for linear hypothesis
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)
# Sum of Squares Error
f.write("Sum of Squared Error 0.5 training\n")
sse_b = 0.5*sum((y_test - Ypred)**2)
# plot2 = plotG(abs(Ypred), y_test, h, '0.5 Training')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_b_lr = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm_b = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np_b = 0.5*sum((numYpred-y_test)**2)
result = pd.DataFrame({"My": sse_b, "SkLearn": sse_b_lr, "numpy": sse_np_b, "Statsmodels": sse_sm_b}, index=[1])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

# Prepare training and test set for C 20%:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=123456)
# w for linear hypothesis
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)
# Sum of Squares Error
f.write("Sum of Squared Error 0.2 training\n")
sse_c = 0.5*sum((y_test - Ypred)**2)
# plot3 = plotG(abs(Ypred), y_test, h, '0.2 Training')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_c_lr = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm_c = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np_c = 0.5*sum((numYpred-y_test)**2)

result = pd.DataFrame({"My": sse_c, "SkLearn": sse_c_lr, "numpy": sse_np_c, "Statsmodels": sse_sm_c}, index=[2])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

sums = [sse_a, sse_b, sse_c]
plot4 = plotS(sums, 'SSE for A, B, C No Normalization')
# pp = PdfPages('myplots_boston.pdf')
# pp.savefig(plot1)
# pp.savefig(plot2)
# pp.savefig(plot3)
pp.savefig(plot4)


# Prepare training and test set for A 80% normalized:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123456)
# normalization
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# w for linear hypothesis
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)

# Sum of Squares Error
f.write("Sum of Squared Error 0.8 training\n")
sse_a_n = 0.5*sum((y_test - Ypred)**2)
# plot5 = plotG(abs(Ypred), y_test, h, '0.8 Training, Normalization')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_a_lr_n = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm_a_n = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np_a_n = 0.5*sum((numYpred-y_test)**2)

result = pd.DataFrame({"My": sse_a_n, "SkLearn": sse_a_lr_n, "numpy": sse_np_a_n, "Statsmodels": sse_sm_a_n}, index=[3])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

# pp.savefig(plot5)

# Prepare training and test set for A 50% Normalized:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=123456)
# normalization
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# w for linear hypothesis
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)

# Sum of Squares Error
f.write("Sum of Squared Error 0.5 training\n")
sse_b_n = 0.5*sum((y_test - Ypred)**2)
# plot6 = plotG(abs(Ypred), y_test, h, '0.5 Training, Normalization')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_b_lr_n = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm_b_n = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np_b_n = 0.5*sum((numYpred-y_test)**2)

result = pd.DataFrame({"My": sse_b_n, "SkLearn": sse_b_lr_n, "numpy": sse_np_b_n, "Statsmodels": sse_sm_b_n}, index=[4])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

# pp.savefig(plot6)


# Prepare training and test set for C 20% Normalized:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=123456)
# normalization
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# w for linear hypothesis
w = lReg(x_train, y_train)
# predicted Y values
Ypred = lPred(w, x_test)

# Sum of Squares Error
f.write("Sum of Squared Error 0.2 training\n")
sse_c_n = 0.5*sum((y_test - Ypred)**2)
# plot7 = plotG(abs(Ypred), y_test, h, '0.2 Training, Normalization')
# Sklearn linear regression test
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_ypred = lr.predict(x_test)
sse_c_lr_n = 0.5*sum((y_test - lr_ypred)**2)
# statsmodel.LR
x_train = sm.add_constant(x_train)
mod = sm.OLS(y_train, x_train)
res = mod.fit()
smyPred = lPred(res.params, x_test)
sse_sm_c_n = 0.5*sum((smyPred-y_test)**2)
# numpy.LG
z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)
numYpred = lPred(z, x_test)
sse_np_c_n = 0.5*sum((numYpred-y_test)**2)

result = pd.DataFrame({"My": sse_c_n, "SkLearn": sse_c_lr_n, "numpy": sse_np_c_n, "Statsmodels": sse_sm_c_n}, index=[5])
results = pd.DataFrame.to_string(result)
f.write(results)
f.write('\n\n')

sums2 = [sse_a, sse_b, sse_c]
plot8 = plotS(sums, 'SSE for A, B, C, Normalization')
# pp.savefig(plot7)
pp.savefig(plot8)

f.close()
pp.close()

# Testing

# Sklearn linear regression test
#lr = LinearRegression()
#lr.fit(x_train, y_train)
# Weights from
#w_lr = lr.coef_
#i_lr = lr.intercept_
#lr_ypred = lr.predict(x_test)
#sse = 0.5*sum((y_test - lr_ypred)**2)
#print("SkLearn, Sum of Squared Error 0.8")
#print(sse)

# statsmodel.LR
#x_train = sm.add_constant(x_train)
#mod = sm.OLS(y_train, x_train)
#res = mod.fit()

# numpy.LG
#z, reis, rank, sigma = np.linalg.lstsq(x_train, y_train)

#result = pd.DataFrame({"My": weights, "SkLearn": np.insert(lr.coef_, 0, lr.intercept_), "numpy": z, "Statsmodels": res.params})
#print(result)
