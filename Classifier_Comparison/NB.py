import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mean

file_output = ["NB_Split.txt", "NB_SKF.txt"]


def main(train, labels, dir_output):

    steps = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    with open(dir_output + file_output[0], "a") as fp:
        fp.write("Size,Ntrial,TestAcc,TrainAcc,ConfMatrix,Config")

    for size in [0.3, 0.5, 0.8]:
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(train, labels, train_size=size,
                                                                random_state=1234 + i)
            for step in steps:
                nb = GaussianNB(var_smoothing=step)
                nb.fit(x_train, y_train)

                y_pred = nb.predict(x_test)
                train_pred = nb.predict(x_train)

                test_acc = accuracy_score(y_test, y_pred)
                train_acc = accuracy_score(y_train, train_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)

                with open(dir_output + file_output[0], "a") as fp:
                    fp.write("\n" + str(size) + "," + str(i) + "," + str(test_acc) + ","
                             + str(train_acc) + "," + str(conf_matrix).replace("\n","") + "," + str(step))

    with open(dir_output + file_output[1], "a") as fp:
        fp.write("Size,Ntrial,TestAcc,TrainAcc,ConfMatrix,Config")

    for size in [3, 7, 10]:
        for i in range(10):
            skf = StratifiedKFold(n_splits=size, random_state=1234 + i)
            X = np.asarray(train)
            Y = np.asarray(labels)

            for step in steps:
                nb = GaussianNB(var_smoothing=step)

                train_acc_fold = []
                test_acc_fold = []
                conf_matrix_fold = []

                for train_index, test_index in skf.split(train, labels):
                    x_train, x_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]

                    nb.fit(x_train, y_train)

                    y_pred = nb.predict(x_test)
                    train_pred = nb.predict(x_train)

                    test_acc_fold.append(accuracy_score(y_test, y_pred))
                    train_acc_fold.append(accuracy_score(y_train, train_pred))
                    conf_matrix_fold.append(confusion_matrix(y_test, y_pred))

                test_acc = mean(test_acc_fold)
                train_acc = mean(train_acc_fold)
                conf_matrix = np.mean(conf_matrix_fold, axis=0)

                with open(dir_output + file_output[1], "a") as fp:
                    fp.write("\n" + str(size) + "," + str(i) + "," + str(test_acc) + ","
                             + str(train_acc) + "," + str(conf_matrix).replace("\n","") + "," + str(step))


