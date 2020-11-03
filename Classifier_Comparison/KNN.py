import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mean

file_output = ["KNN_Split.txt", "KNN_SKF.txt"]


def main(train, labels, dir_output):

    neighbors = [i for i in range(1,31,5)]
    distances = [1, 2, 3, 4, 5]

    with open(dir_output + file_output[0], "a") as fp:
        fp.write("Size,Ntrial,TestAcc,TrainAcc,ConfMatrix,Config")

    for size in [0.3, 0.5, 0.8]:
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(train, labels, train_size=size,
                                                                random_state=1234 + i)
            for n in neighbors:
                for p in distances:
                    kn = KNeighborsClassifier(n_neighbors=n, p=p)
                    kn.fit(x_train, y_train)

                    y_pred = kn.predict(x_test)
                    train_pred = kn.predict(x_train)

                    test_acc = accuracy_score(y_test, y_pred)
                    train_acc = accuracy_score(y_train, train_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)

                    with open(dir_output + file_output[0], "a") as fp:
                        fp.write("\n" + str(size) + "," + str(i) + "," + str(test_acc) + ","
                                 + str(train_acc) + "," + str(conf_matrix).replace("\n","") + "," + str(n)
                                 + "_" + str(p))

    with open(dir_output + file_output[1], "a") as fp:
        fp.write("Size,Ntrial,TestAcc,TrainAcc,ConfMatrix,Config")

    for size in [3, 7, 10]:
        for i in range(10):

            skf = StratifiedKFold(n_splits=size, random_state=1234+i)
            X = np.asarray(train)
            Y = np.asarray(labels)

            for n in neighbors:
                for p in distances:
                    kn = KNeighborsClassifier(n_neighbors=n, p=p)

                    train_acc_fold = []
                    test_acc_fold = []
                    conf_matrix_fold = []

                    for train_index, test_index in skf.split(train, labels):
                        x_train, x_test = X[train_index], X[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]

                        kn.fit(x_train, y_train)

                        y_pred = kn.predict(x_test)
                        train_pred = kn.predict(x_train)

                        test_acc_fold.append(accuracy_score(y_test, y_pred))
                        train_acc_fold.append(accuracy_score(y_train, train_pred))
                        conf_matrix_fold.append(confusion_matrix(y_test, y_pred))

                    test_acc = mean(test_acc_fold)
                    train_acc = mean(train_acc_fold)
                    conf_matrix = np.mean(conf_matrix_fold, axis=0)

                    with open(dir_output + file_output[1], "a") as fp:
                        fp.write("\n" + str(size) + "," + str(i) + "," + str(test_acc) + ","
                                 + str(train_acc) + "," + str(conf_matrix).replace("\n", "") + "," + str(n)
                                 + "_" + str(p))


