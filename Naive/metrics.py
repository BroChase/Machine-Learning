import numpy as np
import matplotlib.pyplot as plt


class metrics:

    def confusionmatrix(self,Y, Y_pred):
        cfm = [0, 0, 0, 0]      # TP, FP, TN, FN
        for i in range(0, len(Y)):
            if (Y[i] == 1 and Y_pred[i][0] == 1):       # TP
                cfm[0] += 1
            elif (Y[i] == 1 and Y_pred[i][0] == 0):     # FP
                cfm[1] += 1
            elif (Y[i] == 0 and Y_pred[i][0] == 0):     # TN
                cfm[2] += 1
            elif (Y[i] == 0 and Y_pred[i][0] == 1):     # FN
                cfm[3] += 1
        return cfm

    def confusionmatrix2(self,Y, Y_pred):
        cfm = [0, 0, 0, 0]      # TP, FP, TN, FN
        for i in range(0, len(Y)):
            if (Y[i] == 1 and Y_pred[i] == 1):       # TP
                cfm[0] += 1
            elif (Y[i] == 1 and Y_pred[i] == 0):     # FP
                cfm[1] += 1
            elif (Y[i] == 0 and Y_pred[i] == 0):     # TN
                cfm[2] += 1
            elif (Y[i] == 0 and Y_pred[i] == 1):     # FN
                cfm[3] += 1
        return cfm

    def precision(self, true_p, false_p):
        if (true_p + false_p) == 0:
            return 0
        else:
            return true_p / (true_p + false_p)

    def accuracy(self, true_p, false_p, true_n, false_n):
        if (true_p + false_p + true_n + false_n) == 0:
            return 0
        else:
            return (true_p + true_n) / (true_p + false_p + true_n + false_n)

    def recall(self, true_p, false_n):
        if (true_p + false_n) == 0:
            return 0
        else:
            return true_p / (true_p + false_n)

    def f1(self, precision, recall):
        if (precision + recall) == 0:
            return 0
        else:
            return 2 * ((precision * recall) / (precision + recall))

    def ROC(self, y_pred, y_test):

        min_score = min(y_pred)
        max_score = max(y_pred)
        # create thresholds space
        thresholds = np.linspace(min_score, max_score, 1000)
        # to hold x and y values
        ROC = np.zeros((1000, 2))

        for index, T in enumerate(thresholds):
            TP = np.logical_and(y_pred > T, y_test == 1).sum()
            TN_t = np.logical_and(y_pred <= T, y_test == 0).sum()
            FP = np.logical_and(y_pred > T, y_test == 0).sum()
            FN_t = np.logical_and(y_pred <= T, y_test == 1).sum()
            ROC[index, 1] = (TP / (TP + FN_t))
            ROC[index, 0] = (FP / (FP + TN_t))
        return ROC

    def gphs(self, ROC, title):
        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.plot(ROC[:, 0], ROC[:, 1], lw=1)
        plt.plot([0, 1], [0, 1], lw=.5, alpha=.6, linestyle='--', color='black')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('$FPR(thresh)$')
        plt.ylabel('$TPR(thresh)$')
        t = title + '.png'
        plt.savefig(t)
        plt.show()

    def gphsall(self, ROC_nb, ROC_bgr, ROC_slr, title):
        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.plot(ROC_nb[:, 0], ROC_nb[:, 1], label='Naive Bayes', lw=1, color='cyan')
        plt.plot(ROC_bgr[:, 0], ROC_bgr[:, 1], label='BGD-LR', lw=1, color='coral')
        plt.plot(ROC_slr[:, 0], ROC_slr[:, 1], label='SGD-LR', lw=1, color='lavender')
        plt.plot([0, 1], [0, 1], lw=.5, alpha=.6, linestyle='--', color='black')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('$FPR(thresh)$')
        plt.ylabel('$TPR(thresh)$')
        plt.legend(loc='best', ncol=3)
        t = title + '.png'
        plt.savefig(t)
        plt.show()