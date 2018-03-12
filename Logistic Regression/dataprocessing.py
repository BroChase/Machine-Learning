# Machine Learning Program 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def sigF(z):
    return 1.0 / (1.0 + np.exp(-z))


def ypred(x_test, weights):
    # x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    ypred = x_test.dot(weights)
    return ypred

def plotS(Averages, title, range, yticks, xticks, ylabel, xlabel):
    fig = plt.figure(figsize=(8, 10), dpi=100)
    plt.title(title)
    plt.bar(range, Averages, 0.35)
    plt.yticks(yticks)
    plt.xticks(x, xticks, rotation='vertical')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return fig

# tolerance conversion: takes the Ypredicted value and uses the sigmoid function to return a value between
# 0 and 1.  anything below .5 is a 0 anything above is a 1.  Replace in ypred


def tolconv(ypred, tol):
    for index, item in enumerate(ypred):
        s = sigF(item)
        if s >= tol:
            ypred[index] = 1
        elif s < tol:
            ypred[index] = 0
    return ypred

# calculate the accuracy precision recall and F1 and prints to a file
# Kfold prints each iteration of the loop with a final average from the Kfold.
# 50/50 prints the scores


def avgmetrics(metrics, k, atmetrics):
    sumaccuracy = 0
    sumprecision = 0
    sumrecall = 0
    sumfone = 0
    iter = 1
    # holds the performance Average of Kfold and performance of 50/50
    performance = []
    if k > 1:
        f.write('K\tAccuracy\tPrecision\tRecall\tF1 \n')
    for i in metrics:
        if k > 1:
            f.write('{:d}'.format(iter))
            f.write('\t{:.3f}'.format(i[0]))
            f.write('\t\t{:.3f}'.format(i[1]))
            f.write('\t\t{:.3f}'.format(i[2]))
            f.write('\t\t{:.3f}\n'.format(i[3]))
        sumaccuracy += i[0]
        sumprecision += i[1]
        sumrecall += i[2]
        sumfone += i[3]
        iter += 1
    if k > 1:
        f.write('\nKfolds Averages\n')
        f.write('Accuracy\tPercision\tRecall\tF1 \n')
        f.write('{:.3f}'.format((sumaccuracy / k)))
        performance.append((sumaccuracy / k))
        f.write('\t\t{: .3f}'.format((sumprecision / k)))
        performance.append((sumprecision / k))
        f.write('\t{: .3f}'.format((sumrecall / k)))
        performance.append((sumrecall / k))
        f.write('\t{: .3f}\n\n'.format((sumfone / k)))
        performance.append((sumfone / k))
    else:
        f.write('Accuracy\tPercision\tRecall\tF1 \n')
        for i in metrics:
            f.write('{:.3f}'.format(i[0]))
            performance.append(i[0])
            f.write('\t\t{: .3f}'.format(i[1]))
            performance.append(i[1])
            f.write('\t{: .3f}'.format(i[2]))
            performance.append(i[2])
            f.write('\t{: .3f}\n\n'.format(i[3]))
            performance.append(i[3])
    # atmetrics holds the performance values from each run to be compared later in a datatable.
    atmetrics.append(performance)
    return

# calculate the Batch Gradient Decent
# alpha = learning rate '0.1, 0.01, 0.001' | nepoch = iterations


def BatchGD(alpha, x, y, nepoch):
    m = x.shape[0]  # number of samples
    theta = np.random.uniform(size=x.shape[1])
    for iter in range(0, nepoch):
        hypothesis = sigF(x.dot(theta))
        error = hypothesis - y
        # J = np.sum(error ** 2) / (2 * m)
        # print("iter %s | J: %.3f" % (iter+1, J))
        gradient = np.dot(x.T, error) / m
        theta = theta - (alpha*gradient)  # update the theta for for next loop
    return theta


def regBatchGD(alpha, x, y, nepoch, lam):
    m = x.shape[0]  # number of samples
    theta = np.random.uniform(size=x.shape[1])
    for iter in range(0, nepoch):
        hypothesis = sigF(x.dot(theta))
        error = hypothesis - y
        # J = np.sum(error ** 2) / (2 * m)
        # print("iter %s | J: %.3f" % (iter+1, J))
        gradient = np.dot(x.T, error)
        theta = theta - (alpha * gradient + (lam/m)*theta)  # update the theta for for next loop
    return theta


def REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, alpha, nepoch, k, atmetrics, lam):
    kf = KFold(n_splits=k)
    metrics = []
    metricssplit = []
    for train_index, test_index in kf.split(dftrain_x):
        # print('train: ', train_index, 'test: ', test_index)
        X_train, X_test = dftrain_x[train_index], dftrain_x[test_index]
        Y_train, Y_test = dftrain_y[train_index], dftrain_y[test_index]
        thetas = regBatchGD(alpha, X_train, Y_train, nepoch, lam)
        y_pred = np.dot(X_test, thetas)
        y_pred = tolconv(y_pred, 0.50)
        # cm = confusion_matrix(Y_test, y_pred)
        #metrics.append(metricscores(cm))
        metrics.append(metricscores(Y_test, y_pred))
    f.write('KFolds cross-validation, K ={:d} alpha={:.4f} nepoch={:d}\n'.format(k, alpha, nepoch))
    avgmetrics(metrics, k, atmetrics)

    # 50% train vs 50% test data sets.
    f.write('50% Train 50% test metrics, alpha={:.4f} nepoch={:d}\n'.format(alpha, nepoch))
    thetas = regBatchGD(alpha, dftrain_x, dftrain_y, nepoch, lam)
    y_pred = np.dot(dftest_x, thetas)
    y_pred = tolconv(y_pred, 0.50)
    # cm = confusion_matrix(dftest_y, y_pred)
    metricssplit.append(metricscores(dftest_y, y_pred))
    k = 1
    avgmetrics(metricssplit, k, atmetrics)
    return


def dataaverages(metrics, averages, precision, recall, fone):
    a_sum = 0
    p_sum = 0
    r_sum = 0
    f_sum = 0
    count = 0
    for index, item in enumerate(metrics):
        a_sum += item[0]
        p_sum += item[1]
        r_sum += item[2]
        f_sum += item[3]
        count += 1
    averages.append(a_sum / count)
    precision.append(p_sum / count)
    recall.append(r_sum / count)
    fone.append(f_sum / count)
    return
# calculate the metric scores accuracy precision recall F1


def metricscores(y_test, y_pred):
    scores = []
    # TN=cm00 | FN=cm10 | TP=cm11 | FP=cm01
    # accuracy (TP+TN) / (TP+TN+FP+FN)
    # accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    # precision (TP) / (TP +FP)
    # if (cm[1][1] + cm[0][1]) == 0:
    #     precision = 1
    # else:
    #     precision = (cm[1][1]) / (cm[1][1] + cm[0][1])
    precision = precision_score(y_test, y_pred)
    scores.append(precision)
    # recall (TP) / (TP + FN)
    # if (cm[1][1] + cm[1][0]) == 0:
    #     recall = 1
    # else:
    #     recall = (cm[1][1]) / (cm[1][1] + cm[1][0])
    recall = recall_score(y_test, y_pred)
    scores.append(recall)
    # f1 2 * (precision * recall) / (precision + recall)
    # if (precision + recall) == 0:
    #     fone = 1
    # fone = 2 * (precision * recall) / (precision + recall)
    fone = f1_score(y_test, y_pred)
    # if the result of fone is nan due to division by zero
    scores.append(fone)
    # print('Confusion Matrix F1: {:.2f}'.format(fone))
    return scores

# Do logistic Regression with the set of perams


def doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, alpha, nepoch, k, atmetrics):
    kf = KFold(n_splits=k)
    metrics = []
    metricssplit = []
    for train_index, test_index in kf.split(dftrain_x):
        # print('train: ', train_index, 'test: ', test_index)
        X_train, X_test = dftrain_x[train_index], dftrain_x[test_index]
        Y_train, Y_test = dftrain_y[train_index], dftrain_y[test_index]
        thetas = BatchGD(alpha, X_train, Y_train, nepoch)
        y_pred = np.dot(X_test, thetas)
        y_pred = tolconv(y_pred, 0.50)
        # cm = confusion_matrix(Y_test, y_pred)
        #metrics.append(metricscores(cm))
        metrics.append(metricscores(Y_test, y_pred))
    f.write('KFolds cross-validation, K ={:d} alpha={:.4f} nepoch={:d}\n'.format(k, alpha, nepoch))
    avgmetrics(metrics, k, atmetrics)

    # 50% train vs 50% test data sets.
    f.write('50% Train 50% test metrics, alpha={:.4f} nepoch={:d}\n'.format(alpha, nepoch))
    thetas = BatchGD(alpha, dftrain_x, dftrain_y, nepoch)
    y_pred = np.dot(dftest_x, thetas)
    y_pred = tolconv(y_pred, 0.50)
    # cm = confusion_matrix(dftest_y, y_pred)
    metricssplit.append(metricscores(dftest_y, y_pred))
    k = 1
    avgmetrics(metricssplit, k, atmetrics)
    return


if __name__ == '__main__':
    f = open('report.doc', 'w')
    pp = PdfPages('myplots.pdf')
    dftrain = pd.read_csv('bank-small-train.csv', sep=';')
    dftest = pd.read_csv('bank-small-test.csv', sep=';')
    for column in dftrain:
        if dftrain[column].dtypes == object:
            dftrain[column] = dftrain[column].astype('category')
            dftrain[column] = dftrain[column].cat.codes
            dftrain[column] = dftrain[column].astype(np.int64)

    for column in dftest:
        if dftest[column].dtypes == object:
            dftest[column] = dftest[column].astype('category')
            dftest[column] = dftest[column].cat.codes
            dftest[column] = dftest[column].astype(np.int64)

    dftrain_x = dftrain.iloc[:, :-1].values
    dftrain_x = np.c_[np.ones(dftrain_x.shape[0]), dftrain_x]
    dftrain_y = dftrain.iloc[:, -1].values

    dftest_x = dftest.iloc[:, :-1].values
    dftest_x = np.c_[np.ones(dftest_x.shape[0]), dftest_x]
    dftest_y = dftest.iloc[:, -1].values

    atmetrics = []
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, atmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, atmetrics)
    # create a datafram with the performance of the above runs and export to csv file
    df = pd.DataFrame(atmetrics)
    df.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    new_col = ['Kfold 0.1, 100', '0.1, 100', 'Kfold 0.01, 100', '0.01, 100', 'Kfold 0.001, 100', '0.001, 100',
               'Kfold 0.1, 500', '0.1, 500', 'Kfold 0.01, 500', '0.01, 500', 'Kfold 0.001, 500', '0.001, 500',
               'Kfold 0.1, 1000', '0.1, 1000', 'Kfold 0.01, 1000', '0.01, 1000', 'Kfold 0.001, 1000', '0.001, 1000']
    df.insert(loc=0, column='Alpha, nepoch', value=new_col)
# *************************************************************************************
# Min max
    f.write('Min-Max scaling results\n')
    dftrain_x = dftrain.iloc[:, :-1].values
    dftrain_y = dftrain.iloc[:, -1].values
    dftest_x = dftest.iloc[:, :-1].values
    dftest_y = dftest.iloc[:, -1].values

    scaler = MinMaxScaler()
    scaler.fit(dftrain_x)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    dftrain_x = scaler.transform(dftrain_x)
    dftrain_x = np.c_[np.ones(dftrain_x.shape[0]), dftrain_x]

    scaler.fit(dftest_x)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    dftest_x = scaler.transform(dftest_x)
    dftest_x = np.c_[np.ones(dftest_x.shape[0]), dftest_x]


    btmetrics = []
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, btmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, btmetrics)
    # create a datafram with the performance of the above runs and export to csv file
    df1 = pd.Series([], name='Min-Max')
    df2 = pd.DataFrame(btmetrics)
    df2.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df2.insert(loc=0, column='Alpha, nepoch', value=new_col)

# ***********************************************************************
# Standardization
    f.write('Standardization results\n')
    dftrain_x = dftrain.iloc[:, :-1].values
    dftrain_y = dftrain.iloc[:, -1].values
    dftest_x = dftest.iloc[:, :-1].values
    dftest_y = dftest.iloc[:, -1].values

    scaler = StandardScaler()

    dftrain_x = scaler.fit_transform(dftrain_x)
    dftrain_x = np.c_[np.ones(dftrain_x.shape[0]), dftrain_x]

    dftest_x = scaler.fit_transform(dftest_x)
    dftest_x = np.c_[np.ones(dftest_x.shape[0]), dftest_x]

    ctmetrics = []
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, ctmetrics)
    doregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, ctmetrics)
    # create a datafram with the performance of the above runs and export to csv file
    df3 = pd.Series([], name='Standardization')
    df4 = pd.DataFrame(ctmetrics)
    df4.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df4.insert(loc=0, column='Alpha, nepoch', value=new_col)

# *********************************************************************
# Regularized logistic regression with Standarization Xtrain Xtest lambda = 0
    f.write('Regularized Logistic Regression with Standardization data: Lambda = 0 \n')
    dtmetrics = []
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, dtmetrics, lam=0)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, dtmetrics, lam=0)
    df5 = pd.Series([], name='Regularized LG, Lambda=0')
    df6 = pd.DataFrame(dtmetrics)
    df6.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df6.insert(loc=0, column='Alpha, nepoch', value=new_col)
# ***************************************************
# lambda = 1
    f.write('Regularized Logistic Regression with Standardization data: Lambda = 1 \n')
    etmetrics = []
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, etmetrics, lam=1)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, etmetrics, lam=1)
    df7 = pd.Series([], name='Regularized LG, Lambda=1')
    df8 = pd.DataFrame(etmetrics)
    df8.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df8.insert(loc=0, column='Alpha, nepoch', value=new_col)
#************************************************************
# lambda = 10
    f.write('Regularized Logistic Regression with Standardization data: Lambda = 10 \n')
    ftmetrics = []
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, ftmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, ftmetrics, lam=10)
    df9 = pd.Series([], name='Regularized LG, Lambda=10')
    df10 = pd.DataFrame(ftmetrics)
    df10.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df10.insert(loc=0, column='Alpha, nepoch', value=new_col)
#***************************************************************
# lambda = 100
    f.write('Regularized Logistic Regression with Standardization data: Lambda = 100 \n')
    gtmetrics = []
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, gtmetrics, lam=10)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, gtmetrics, lam=100)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, gtmetrics, lam=100)
    df11 = pd.Series([], name='Regularized LG, Lambda=100')
    df12 = pd.DataFrame(gtmetrics)
    df12.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df12.insert(loc=0, column='Alpha, nepoch', value=new_col)
# ************************************************************
# lambda = 1000
    f.write('Regularized Logistic Regression with Standardization data: Lambda = 1000 \n')
    htmetrics = []
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 100, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 100, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 100, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 500, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 500, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 500, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.1, 1000, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.01, 1000, 10, htmetrics, lam=1000)
    REGdoregression(dftrain_x, dftrain_y, dftest_x, dftest_y, 0.001, 1000, 10, htmetrics, lam=1000)
    df13 = pd.Series([], name='Regularized LG, Lambda=1000')
    df14 = pd.DataFrame(htmetrics)
    df14.columns = ['Accuracy', 'Percision', 'Recall', 'F1']
    df14.insert(loc=0, column='Alpha, nepoch', value=new_col)
    result = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14], axis=1)
# ***************************************************
# Plot Accuracy Averages from each set
    Averages = []
    Precision = []
    Recall = []
    FOne = []
    dataaverages(atmetrics, Averages, Precision, Recall, FOne)
    dataaverages(btmetrics, Averages, Precision, Recall, FOne)
    dataaverages(ctmetrics, Averages, Precision, Recall, FOne)
    dataaverages(dtmetrics, Averages, Precision, Recall, FOne)
    dataaverages(etmetrics, Averages, Precision, Recall, FOne)
    dataaverages(ftmetrics, Averages, Precision, Recall, FOne)
    dataaverages(gtmetrics, Averages, Precision, Recall, FOne)
    dataaverages(htmetrics, Averages, Precision, Recall, FOne)
    x = np.arange(8)
    title = 'Accuracy Performance Averages'
    yticks = np.arange(0, 1, 0.05)
    ylabel = 'Performance'
    xlabel = 'Params'
    xticks = ['Pure', 'MinMax', 'Standardized', 'Regularized', 'Lambda0', 'Lambda1', 'Lambda10', 'Lambda100', 'Lambda1000']
    plot1 = plotS(Averages, title, x, yticks, xticks, ylabel, xlabel)
    title = 'Precision Performance Averages'
    yticks = np.arange(0, 0.50, 0.02)
    plot2 = plotS(Precision, title, x, yticks, xticks, ylabel, xlabel)
    title = 'Recall Performance Averages'
    plot3 = plotS(Recall, title, x, yticks, xticks, ylabel, xlabel)
    title = 'F1 Performance Averages'
    plot4 = plotS(FOne, title, x, yticks, xticks, ylabel, xlabel)
    pp.savefig(plot1)
    pp.savefig(plot2)
    pp.savefig(plot3)
    pp.savefig(plot4)
    result.to_csv('Table.csv', sep=',')
    pp.close()
    f.close()