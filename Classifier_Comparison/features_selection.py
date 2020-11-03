import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import warnings
from statistics import mean
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


with open("./dataset/spiral_full.txt", "r") as fp:
    lines_spiral = fp.readlines()[1:]

train_spiral = []
labels_spiral = []

for line in lines_spiral:
    train_spiral.append([float(x) for x in line.split('\t')[:-1]])
    labels_spiral.append(float(line[-2]))

with open("./dataset/meander_full.txt", "r") as fp:
    lines_meander = fp.readlines()[1:]

train_meander = []
labels_meander = []

for line in lines_meander:
    train_meander.append([float(x) for x in line.split('\t')[:-1]])
    labels_meander.append(float(line[-2]))

X = np.asarray(train_spiral)
Y = np.asarray(labels_spiral)
features_scores_1 = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0],
                     5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0]}

for rnd in range(10):
    skf = StratifiedKFold(n_splits=7, random_state=1234 + rnd)

    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # SelectKBest
        test = SelectKBest(score_func=chi2, k=9)
        fit = test.fit(x_train, y_train)
        f = list(fit.scores_)
        sort_f = sorted(range(len(f)), key=lambda k: f[k], reverse=True)
        for i in range(len(sort_f)):
            features_scores_1[sort_f[i]][0] += 8 - i

        # RFE
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(x_train, y_train)
        # sort2 = sorted(range(len(list(fit.ranking_))), key=lambda k: list(fit.ranking_)[k])
        # print("Feature Ranking: %s" % sort2)
        for i in range(len(list(fit.ranking_))):
            features_scores_1[i][1] += list(fit.ranking_)[i]

        # PCA
        model = PCA()
        fit = model.fit(x_train)
        n_pcs = model.components_.shape[0]
        most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
        for i in range(len(list(most_important))):
            features_scores_1[list(most_important)[i]][2] += 8 - i

for i in range(9):
    features_scores_1[i] = [round(v/70,2) for v in features_scores_1[i]]
sort = sorted(features_scores_1, key=features_scores_1.get, reverse=True)
for i in sort:
    print(i, features_scores_1[i])

print("****************************************************")

features_scores_2 = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0],
                     5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0]}

X = np.asarray(train_meander)
Y = np.asarray(labels_meander)

for rnd in range(10):
    skf = StratifiedKFold(n_splits=7, random_state=1234 + rnd)

    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # SelectKBest
        test = SelectKBest(score_func=chi2, k=9)
        fit = test.fit(x_train, y_train)
        f = list(fit.scores_)
        sort_f = sorted(range(len(f)), key=lambda k: f[k], reverse=True)
        for i in range(len(sort_f)):
            features_scores_2[sort_f[i]][0] += 8 - i

        # RFE
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(x_train, y_train)
        # sort2 = sorted(range(len(list(fit.ranking_))), key=lambda k: list(fit.ranking_)[k])
        # print("Feature Ranking: %s" % sort2)
        for i in range(len(list(fit.ranking_))):
            features_scores_2[i][1] += 8 - list(fit.ranking_)[i]

        # PCA
        model = PCA()
        fit = model.fit(x_train)
        n_pcs = model.components_.shape[0]
        most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
        for i in range(len(list(most_important))):
            features_scores_2[list(most_important)[i]][2] += 8 - i

for i in range(9):
    features_scores_2[i] = [round(v/70,2) for v in features_scores_2[i]]

sort_2 = sorted(features_scores_2, key=features_scores_2.get, reverse=True)
for i in sort_2:
    print(i, features_scores_2[i])

print("****************************************************")

features_scores_3 = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0],
                     5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0], 8: [0, 0, 0]}

X = np.asarray(train_spiral + train_meander)
Y = np.asarray(labels_spiral + labels_meander)

for rnd in range(10):
    skf = StratifiedKFold(n_splits=7, random_state=1234 + rnd)

    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # SelectKBest
        test = SelectKBest(score_func=chi2, k=9)
        fit = test.fit(x_train, y_train)
        f = list(fit.scores_)
        sort_f = sorted(range(len(f)), key=lambda k: f[k], reverse=True)
        for i in range(len(sort_f)):
            features_scores_3[sort_f[i]][0] += 8 - i

        # RFE
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model)
        fit = rfe.fit(x_train, y_train)
        # sort2 = sorted(range(len(list(fit.ranking_))), key=lambda k: list(fit.ranking_)[k])
        # print("Feature Ranking: %s" % sort2)
        for i in range(len(list(fit.ranking_))):
            features_scores_3[i][1] += 8 - list(fit.ranking_)[i]

        # PCA
        model = PCA()
        fit = model.fit(x_train)
        n_pcs = model.components_.shape[0]
        most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
        for i in range(len(list(most_important))):
            features_scores_3[list(most_important)[i]][2] += 8 - i

for i in range(9):
    features_scores_3[i] = [round(v/70,2) for v in features_scores_3[i]]

sort_3 = sorted(features_scores_3, key=features_scores_3.get, reverse=True)
for i in sort_3:
    print(i, features_scores_3[i])

print("****************************************************")

l1 = [0,0,0,0,0,0,0,0,0]
l2 = [0,0,0,0,0,0,0,0,0]
l3 = [0,0,0,0,0,0,0,0,0]

for i in range(9):
    l1[i] = mean(features_scores_1[i])
sort = sorted(range(len(l1)), key=lambda k: l1[k], reverse=True)
print(sort)


for i in range(9):
    l2[i] = mean(features_scores_2[i])
sort = sorted(range(len(l2)), key=lambda k: l2[k], reverse=True)
print(sort)

for i in range(9):
    l3[i] = mean(features_scores_3[i])
sort = sorted(range(len(l3)), key=lambda k: l3[k], reverse=True)
print(sort)

