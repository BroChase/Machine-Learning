import dataproc
import nbclassifier
import BGlogisticReg
import stochasticlogicregression
import metrics
from collections import Counter


import time

if __name__ == '__main__':

    f = open('report.doc', 'w')
    metrics = metrics.metrics()
    print('Loading data...')
    # for loading face and non-face change the path to the appropriate pathing
    face = 'faces/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/train/face'  # 2429 images
    non_face = 'faces/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/train/non-face'  # 4548 img images
    df = dataproc.createdataframe(face, non_face)
    X_train = df.iloc[:, :-1].values
    Y_train = df.iloc[:, -1].values

    face = 'faces/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/test/face'  # 472 images
    non_face = 'faces/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/test/non-face'  # 23573 images
    df2 = dataproc.createdataframe(face, non_face)
    X_test = df2.iloc[:, :-1].values
    Y_test = df2.iloc[:, -1].values
    print('Data Loaded..\n')

    val_dict = dict(Counter(Y_test))
    # Train the Naive Bayes from the Train Dataset 2429 faces and 4548 non-faces
    print('Training Naive Bayes Classifier')
    start_time_nb = time.process_time()
    start_time = time.process_time()
    Naive = nbclassifier.NaiveBayesClassifier()
    Naive.train(X_train, Y_train)
    stop_time = time.process_time()
    print('Naive Bayes Classifier training complete in {:.2f}\n'.format(stop_time-start_time))
    f.write('Naive Bayes Classifier training complete in {:.2f}\n'.format(stop_time-start_time))
    start_time = time.process_time()
    print('Finding Predicted Y Values...')
    Y_pred, roc_preds = Naive.image_classifer(X_test)
    roc_nb = metrics.ROC(roc_preds, Y_test)
    metrics.gphs(roc_nb, 'Naive Bayes')
    stop_time = time.process_time()
    stop_time_nb = time.process_time()
    print('Y Predictions Calculated finished {:.2f} \nTotal elapsed time for Naive Bayes {:.2f}'.format(stop_time-start_time, stop_time_nb-start_time_nb))
    f.write('Y Predictions Calculated finished {:.2f} \nTotal elapsed time for Naive Bayes {:.2f}\n'.format(stop_time - start_time, stop_time_nb - start_time_nb))
    cfm = metrics.confusionmatrix(Y_test, Y_pred)
    nb_accuracy = metrics.accuracy(cfm[0], cfm[1], cfm[2], cfm[3])
    nb_precision = metrics.precision(cfm[0], cfm[1])
    nb_recall = metrics.recall(cfm[0], cfm[3])
    nb_f1 = metrics.f1(nb_precision, nb_recall)
    np_TPR = cfm[0] / val_dict[1]
    np_FPR = cfm[1] / val_dict[0]

    print('Naive Bayes Accuracy {:.2f}'.format(nb_accuracy))
    print('Naive Bayes Precision {:.2f}'.format(nb_precision))
    print('Naive Bayes Recall {:.2f}'.format(nb_recall))
    print('Naive Bayes F1 {:.2f}'.format(nb_f1))
    print('Naive Bayes True positives:{} Out of {} positives TPR:{:.2f}'.format(cfm[0], val_dict[1], np_TPR))
    print('Naive Bayes False Positives:{} Out of {} negatives FPR:{:.2f}\n'.format(cfm[1], val_dict[0], np_FPR))
    f.write('Naive Bayes Accuracy {:.2f}\n'.format(nb_accuracy))
    f.write('Naive Bayes Precision {:.2f}\n'.format(nb_precision))
    f.write('Naive Bayes Recall {:.2f}\n'.format(nb_recall))
    f.write('Naive Bayes F1 {:.2f}\n'.format(nb_f1))
    f.write('Naive Bayes True positives:{} Out of {} positives TPR:{:.2f}\n'.format(cfm[0], val_dict[1], np_TPR))
    f.write('Naive Bayes False Positives:{} Out of {} negatives FPR:{:.2f}\n\n\n'.format(cfm[1], val_dict[0], np_FPR))



    start_time = time.process_time()
    BGR = BGlogisticReg.LogisticRegression()
    thetas = BGR.regBatchGD(0.5, X_train, Y_train, 1000, 1)
    y_pred = BGR.y_pred(X_test, thetas)
    roc_bgr = metrics.ROC(y_pred, Y_test)  # Logistical
    metrics.gphs(roc_bgr, 'BGD-LR')
    y_pred = BGR.tolconv(y_pred, 0.50)
    stop_time = time.process_time()
    print('Total elapsed time for Batch Gradient Logistic Regression {:.2f}'.format(stop_time - start_time))
    f.write('Total elapsed time for Batch Gradient Logistic Regression {:.2f}\n'.format(stop_time - start_time))
    cfm2 = metrics.confusionmatrix2(Y_test, y_pred)
    lr_accuracy = metrics.accuracy(cfm2[0], cfm2[1], cfm2[2], cfm2[3])
    lr_precision = metrics.precision(cfm2[0], cfm2[1])
    lr_recall = metrics.recall(cfm2[0], cfm2[3])
    lr_f1 = metrics.f1(lr_precision, lr_recall)
    lr_TPR = cfm2[0] / val_dict[1]
    lr_FPR = cfm2[1] / val_dict[0]

    print('Batch Gradient Logistic Regression Accuracy {:.2f}'.format(lr_accuracy))
    print('Batch Gradient Logistic Regression Precision {:.2f}'.format(lr_precision))
    print('Batch Gradient Logistic Regression Recall {:.2f}'.format(lr_recall))
    print('Batch Gradient Logistic Regression F1 {:.2f}'.format(lr_f1))
    print('Batch Gradient Logistic Regression True positives:{} Out of {} positives TPR:{:.2f}'.format(cfm2[0], val_dict[1], lr_TPR))
    print('Batch Gradient Logistic Regression False Positives:{} Out of {} negatives FPR:{:.2f}\n'.format(cfm2[1], val_dict[0], lr_FPR))
    f.write('Batch Gradient Logistic Regression Accuracy {:.2f}\n'.format(lr_accuracy))
    f.write('Batch Gradient Logistic Regression Precision {:.2f}\n'.format(lr_precision))
    f.write('Batch Gradient Logistic Regression Recall {:.2f}\n'.format(lr_recall))
    f.write('Batch Gradient Logistic Regression F1 {:.2f}\n'.format(lr_f1))
    f.write('Batch Gradient Logistic Regression True positives:{} Out of {} positives TPR:{:.2f}\n'.format(cfm2[0],
                                                                                                       val_dict[1],
                                                                                                       lr_TPR))
    f.write('Batch Gradient Logistic Regression False Positives:{} Out of {} negatives FPR:{:.2f}\n\n\n'.format(cfm2[1],
                                                                                                          val_dict[0],
                                                                                                          lr_FPR))

    start_time = time.process_time()
    SLR = stochasticlogicregression.StochLogisticRegression()
    thetas = SLR.sgd_logistic(X_train, Y_train, 1000, .5, 1)
    slr_y_pred = BGR.y_pred(X_test, thetas)
    roc_slr = metrics.ROC(slr_y_pred, Y_test)     # Stochastic
    metrics.gphs(roc_slr, 'SGD-LR')
    slr_y_pred = BGR.tolconv(slr_y_pred, 0.50)
    stop_time = time.process_time()
    print('Total elapsed time for Stochastic Gradient Logistic Regression {:.2f}'.format(stop_time-start_time))
    f.write('Total elapsed time for Stochastic Gradient Logistic Regression {:.2f}\n'.format(stop_time - start_time))
    cfm3 = metrics.confusionmatrix2(Y_test, slr_y_pred)
    slr_accuracy = metrics.accuracy(cfm3[0], cfm3[1], cfm3[2], cfm3[3])
    slr_precision = metrics.precision(cfm3[0], cfm3[1])
    slr_recall = metrics.recall(cfm3[0], cfm3[3])
    slr_f1 = metrics.f1(slr_precision, slr_recall)
    slr_TPR = cfm3[0] / val_dict[1]
    slr_FPR = cfm3[1] / val_dict[0]

    print('Stochastic Gradient Logistic Regression Accuracy {:.2f}'.format(slr_accuracy))
    print('Stochastic Gradient Logistic Regression Precision {:.2f}'.format(slr_precision))
    print('Stochastic Gradient Logistic Regression Recall {:.2f}'.format(slr_recall))
    print('Stochastic Gradient Logistic Regression F1 {:.2f}'.format(slr_f1))
    print('Stochastic Gradient Logistic Regression True positives:{} Out of {} positives TPR:{:.2f}'.format(cfm3[0], val_dict[1], slr_TPR))
    print('Stochastic Gradient Logistic Regression False Positives:{} Out of {} negatives FPR:{:.2f}\n'.format(cfm3[1], val_dict[0], slr_FPR))
    f.write('Stochastic Gradient Logistic Regression Accuracy {:.2f}\n'.format(slr_accuracy))
    f.write('Stochastic Gradient Logistic Regression Precision {:.2f}\n'.format(slr_precision))
    f.write('Stochastic Gradient Logistic Regression Recall {:.2f}\n'.format(slr_recall))
    f.write('Stochastic Gradient Logistic Regression F1 {:.2f}\n'.format(slr_f1))
    f.write('Stochastic Gradient Logistic Regression True positives:{} Out of {} positives TPR:{:.2f}\n'.format(cfm3[0], val_dict[1], slr_TPR))
    f.write('Stochastic Gradient Logistic Regression False Positives:{} Out of {} negatives FPR:{:.2f}\n'.format(cfm3[1], val_dict[0], slr_FPR))

    f.write('\t\tAccuracy\tPrecision\tRecall\tF1\n')
    f.write('Nb\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n'.format(nb_accuracy, nb_precision, nb_recall, nb_f1))
    f.write('BGF-LR\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n'.format(lr_accuracy, lr_precision, lr_recall, lr_f1))
    f.write('SGD-LR\t{:.2f}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}\n'.format(slr_accuracy, slr_precision, slr_recall, slr_f1))

    # graph all three plots together
    metrics.gphsall(roc_nb, roc_bgr, roc_slr, 'NB, BGD-LR, SGD-LR')
    f.close()
