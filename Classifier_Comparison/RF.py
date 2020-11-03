import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statistics import mean
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

file_output = ["RF_Split.txt", "RF_SKF.txt"]


def main(train, labels, dir_output):

    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100]
    max_depths = [None] + [x for x in range(1, 31, 5)]
    min_samples_splits = np.linspace(0.1, 1.0, 5, endpoint=True)
    min_samples_leafs = np.linspace(0.1, 0.5, 3, endpoint=True)
    max_features = [None, 'sqrt', 'log2']

    with open(dir_output + file_output[0], "a") as fp:
        fp.write("Size,Ntrial,AvgTestAcc,AvgTrainAcc,AvgConfMatrix,Config")

    for size in [0.3, 0.5, 0.8]:
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(train, labels, train_size=size, random_state=1234+i)

            for n_estimator in n_estimators:
                for max_depth in max_depths:
                    for min_samples_split in min_samples_splits:
                        for min_samples_leaf in min_samples_leafs:
                            for max_feature in max_features:
                                train_acc = []
                                test_acc = []
                                conf_matrix = []
                                for k in range(5):
                                    work = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                            max_features=max_feature, random_state=1234 + k)
                                    work.fit(x_train, y_train)

                                    y_pred = work.predict(x_test)
                                    train_pred = work.predict(x_train)

                                    test_acc.append(accuracy_score(y_test, y_pred))
                                    train_acc.append(accuracy_score(y_train, train_pred))
                                    conf_matrix.append(confusion_matrix(y_test, y_pred))

                                with open(dir_output + file_output[0], "a") as fp:
                                    fp.write("\n" + str(size) + "," + str(i) + "," + str(mean(test_acc)) + ","
                                             + str(mean(train_acc)) + ","
                                             + str(np.mean(conf_matrix, axis=0)).replace("\n", "") + ","
                                             + str(n_estimator) + "_" + str(max_depth) + "_" + str(min_samples_leaf)
                                                  + "_" + str(min_samples_split) + "_" + str(max_feature))

    with open(dir_output + file_output[1], "a") as fp:
        fp.write("Size,Ntrial,AvgTestAcc,AvgTrainAcc,AvgConfMatrix,Config")

    for size in [3, 7, 10]:
        for i in range(10):

            skf = StratifiedKFold(n_splits=size, random_state=1234+i)
            X = np.asarray(train)
            Y = np.asarray(labels)

            for n_estimator in n_estimators:
                for max_depth in max_depths:
                    for min_samples_split in min_samples_splits:
                        for min_samples_leaf in min_samples_leafs:
                            for max_feature in max_features:
                                train_acc = []
                                test_acc = []
                                conf_matrix = []
                                for k in range(5):
                                    work = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth,
                                                                  min_samples_leaf=min_samples_leaf,
                                                                  min_samples_split=min_samples_split,
                                                                  max_features=max_feature, random_state=1234 + k)
                                    train_acc_fold = []
                                    test_acc_fold = []
                                    conf_matrix_fold = []

                                    for train_index, test_index in skf.split(train, labels):
                                        x_train, x_test = X[train_index], X[test_index]
                                        y_train, y_test = Y[train_index], Y[test_index]

                                        work.fit(x_train, y_train)

                                        y_pred = work.predict(x_test)
                                        train_pred = work.predict(x_train)

                                        test_acc_fold.append(accuracy_score(y_test, y_pred))
                                        train_acc_fold.append(accuracy_score(y_train, train_pred))
                                        conf_matrix_fold.append(confusion_matrix(y_test, y_pred))

                                    test_acc.append(mean(test_acc_fold))
                                    train_acc.append(mean(train_acc_fold))
                                    conf_matrix.append(np.mean(np.mean(conf_matrix_fold, axis=0)))

                                with open(dir_output + file_output[1], "a") as fp:
                                    fp.write("\n" + str(size) + "," + str(i) + "," + str(mean(test_acc)) + ","
                                             + str(mean(train_acc)) + ","
                                             + str(np.mean(conf_matrix, axis=0)).replace("\n", "") + ","
                                             + str(n_estimator) + "_" + str(max_depth) + "_" + str(min_samples_leaf)
                                             + "_" + str(min_samples_split) + "_" + str(max_feature))

