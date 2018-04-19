import dataproc
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


if __name__ == '__main__':
    f = open('report.doc', 'w')
    print('Loading fingerprint data...')

    fingerprints = 'training'
    # change path to change test set.
    testfingers = 'testA'

    df = dataproc.createdataframe(fingerprints)
    print('Finterprints Loaded')
    dfset_y = df.iloc[:, -1].values
    dfset_x = df.iloc[:, :-1].values
    dfset_x = pd.DataFrame(dfset_x)

    # Applying normalization by scaling all values in the frame to a generic darkness
    # Use StandardScaler() to normalize data more
    print('Applying normalization to fingerprints')
    dfset_x = dfset_x.apply(lambda x: x/255)
    u = dfset_x.sum().sum() / (dfset_x.shape[0] * dfset_x.shape[1])
    dfset_x = dfset_x.apply(lambda x: x-u)
    scaler = StandardScaler()
    dfset_x = scaler.fit_transform(dfset_x)
    print('Normalization complete')

    # Classifier is a one vs rest classifier with an rbf kernel.  Memory cache size 1000mb
    clf = OneVsRestClassifier(SVC(kernel='rbf', cache_size=1000))

    # Attempt to find the best C param for more accurate predictions.
    print('Searching for best C params')
    # C_range = [.001, .01, .1, 1, 10, 100]
    C_range = [100]
    param_grid = {'estimator__C': C_range}
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(dfset_x, dfset_y)
    print('Best C Params: ', grid_search.best_params_)

    # Load test set.
    print('Loading dataset: ', testfingers)
    df = dataproc.createdataframe(testfingers)
    print(testfingers, ' loaded')
    dftest_y = df.iloc[:, -1].values
    dftest_x = df.iloc[:, :-1].values

    # Normalize test set in same way as train set
    print('Applying normalization to fingerprints')
    dftest_x = pd.DataFrame(dftest_x)
    dftest_x = dftest_x.apply(lambda x: x/255)
    u = dftest_x.sum().sum() / (dftest_x.shape[0] * dftest_x.shape[1])
    dftest_x = dftest_x.apply(lambda x: x-u)
    dftest_x = scaler.fit_transform(dftest_x)
    print('Normalization complete')

    # Get predictions.
    y_pred = grid_search.predict(dftest_x)

    for i in range(y_pred.shape[-1]):
        print('Fingerprint actual: {:.0f}  Predicted as {:.0f}'.format(dftest_y.item(i), y_pred.item(i)))
        f.write('Fingerprint actual: {:.0f}  Predicted as {:.0f}\n'.format(dftest_y.item(i), y_pred.item(i)))


    print("Model Accuracy: {:.2f}%".format(grid_search.score(dftest_x, dftest_y) * 100))
    f.write("Model Accuracy: {:.2f}%\n".format(grid_search.score(dftest_x, dftest_y) * 100))

    f.close()

