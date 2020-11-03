import numpy as np
from sklearn.svm import SVC
from statistics import mean
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

file_output = ["SVM_Split.txt", "SVM_SKF.txt"]


def main(train, labels, dir_output):

    kernels = ['linear','poly','rbf']
    cs = [0.1, 1, 10, 100, 1000]
    gammas = []
    degrees = []

    with open(dir_output + file_output[0], "a") as fp:
        fp.write("Size,Ntrial,AvgTestAcc,AvgTrainAcc,AvgConfMatrix,Config")

    for size in [0.3, 0.5, 0.8]:
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(train, labels, train_size=size, random_state=1234+i)

            for kernel in kernels:
                for C in cs:
                    if kernel == 'linear':
                        degrees = [3]
                        gammas = ['scale']
                    if kernel == 'poly':
                        degrees = [1, 2, 3, 4, 5, 6]
                        gammas = ['scale']
                    if kernel == 'rbf':
                        degrees = [3]
                        gammas = [1e-9, 1e-5, 0.1, 1, 10, 100]
                    for gamma in gammas:
                        for degree in degrees:
                            train_acc = []
                            test_acc = []
                            conf_matrix = []
                            for k in range(5):
                                work = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, random_state=1234 + k)
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
                                         + str(kernel) + "_" + str(gamma) + "_" + str(C) + "_" + str(degree))

    with open(dir_output + file_output[1], "a") as fp:
        fp.write("Size,Ntrial,AvgTestAcc,AvgTrainAcc,AvgConfMatrix,Config")

    for size in [3, 7, 10]:
        for i in range(10):

            skf = StratifiedKFold(n_splits=size, random_state=1234+i)
            X = np.asarray(train)
            Y = np.asarray(labels)

            for kernel in kernels:
                for C in cs:
                    if kernel == 'linear':
                        degrees = [3]
                        gammas = ['scale']
                    if kernel == 'poly':
                        degrees = [1, 2, 3, 4, 5, 6]
                        gammas = ['scale']
                    if kernel == 'rbf':
                        degrees = [3]
                        gammas = [1e-9, 1e-5, 0.1, 1, 10, 100]
                    for gamma in gammas:
                        for degree in degrees:
                            train_acc = []
                            test_acc = []
                            conf_matrix = []
                            for k in range(5):
                                work = SVC(kernel=kernel, gamma=gamma, C=C, degree=degree, random_state=1234 + k)
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
                                         + str(kernel) + "_" + str(gamma) + "_" + str(C) + "_" + str(degree))

