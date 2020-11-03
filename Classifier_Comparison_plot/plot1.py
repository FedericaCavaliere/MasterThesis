import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np
from math import ceil

classifiers = ["DT", "NB", "KNN", "SVM", "AB", "RF", "MLP"]
datasets = ["spiral", "meander"]
classifier_conf = {"DT": "[max_depth]_[min_samples_split]_[min_samples_leaf]",
                   "NB": "[var_smoothing]", "KNN": "[neighbor]_[distance]", "SVM": "[kernel]_[gamma]_[C]_[degree]",
                   "AB": "[learning_rate]_[n_estimator]_[algorithm]",
                   "RF": "[n_estimator]_[max_depth]_[min_samples_leaf]_[min_samples_split]_[max_feature]",
                   "MLP": "[activation]_[solver]_[learning_rate]_[hidden_layer_size]"}

for dataset in datasets:
    split = {}
    skf = {}

    for classifier in classifiers:
        with open("./result/" + dataset + "/" + classifier + "_Split.txt", 'r') as fp:
            split[classifier] = fp.readlines()

        with open("./result/" + dataset + "/" + classifier + "_SKF.txt", 'r') as fp:
            skf[classifier] = fp.readlines()

        config1 = {}
        config2 = {}
        config3 = {}

        for line in split[classifier][1:]:
            if classifier == "MLP":
                key = line.split(",")[-2].replace("\n", "") + line.split(",")[-1].replace("\n", "")
            else:
                key = line.split(",")[-1].replace("\n", "")

            newline = line.split(",")

            if newline[0] == '0.3':
                config1.setdefault(key, []).append(float(newline[2]))
            elif newline[0] == '0.5':
                config2.setdefault(key, []).append(float(newline[2]))
            else:
                config3.setdefault(key, []).append(float(newline[2]))

        conf_avg_split = {}
        conf_std_split = {}

        for i in config1:
            conf_avg_split[i] = [round(mean(config3[i]), 2), round(mean(config2[i]), 2), round(mean(config1[i]), 2)]
            conf_std_split[i] = [round(stdev(config3[i]), 2), round(stdev(config2[i]), 2), round(stdev(config1[i]), 2)]

        sort = sorted(conf_avg_split, key=conf_avg_split.get)
        x = [i for i in sort]

        plt.plot(x, [conf_avg_split[i][2] for i in sort], color='brown', label='Train = 30%\nmax: '
                                                                + str(max([conf_avg_split[i][2] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_split[i][2] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_split[i][2] for i in sort]) - (np.array([conf_std_split[i][2] for i in sort]))/10,
                         np.array([conf_avg_split[i][2] for i in sort]) + (np.array([conf_std_split[i][2] for i in sort]))/10,
                         alpha=0.2, color='brown')

        plt.plot(x, [conf_avg_split[i][1] for i in sort], color='olivedrab', label='Train = 50%\nmax: '
                                                                + str(max([conf_avg_split[i][1] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_split[i][1] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_split[i][1] for i in sort]) - (np.array([conf_std_split[i][1] for i in sort]))/10,
                         np.array([conf_avg_split[i][1] for i in sort]) + (np.array([conf_std_split[i][1] for i in sort]))/10,
                         alpha=0.2, color='olivedrab')

        plt.plot(x, [conf_avg_split[i][0] for i in sort], color='royalblue', label='Train = 80%\nmax: '
                                                                + str(max([conf_avg_split[i][0] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_split[i][0] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_split[i][0] for i in sort]) - (np.array([conf_std_split[i][0] for i in sort]))/10,
                         np.array([conf_avg_split[i][0] for i in sort]) + (np.array([conf_std_split[i][0] for i in sort]))/10,
                         alpha=0.2, color='royalblue')

        plt.ylabel('Test Accuracy')
        plt.xlabel('Classifier Configuration')
        plt.legend(fontsize=12)
        plt.ylim([0.45, 0.95])
        plt.grid()
        index = np.linspace(0, len(sort) - 1, 8, endpoint=True)
        plt.xticks([sort[ceil(i)] for i in index], rotation=10,fontsize=10)
        plt.title("Dataset Split: " + classifier + " Classifier\n" + "Configuration: " + classifier_conf[classifier], fontweight="bold", fontsize=14)
        ax = plt.gca()
        ax.tick_params(width=2)
        fig = plt.gcf()
        fig.set_size_inches((16, 9), forward=False)
        fig.savefig("./plot/" + dataset + "/single/split/" + classifier + '.png', dpi=500)
        #plt.show()
        plt.clf()

        config4 = {}
        config5 = {}
        config6 = {}

        for line in skf[classifier][1:]:
            if classifier == "MLP":
                key = line.split(",")[-2].replace("\n", "") + line.split(",")[-1].replace("\n", "")
            else:
                key = line.split(",")[-1].replace("\n", "")

            newline = line.split(",")

            if newline[0] == '3':
                config4.setdefault(key, []).append(round(float(newline[2]), 2))
            elif newline[0] == '7':
                config5.setdefault(key, []).append(round(float(newline[2]), 2))
            else:
                config6.setdefault(key, []).append(round(float(newline[2]), 2))

        conf_avg_skf = {}
        conf_std_skf = {}

        for i in config4:
            conf_avg_skf[i] = [round(mean(config6[i]), 2), round(mean(config5[i]), 2), round(mean(config4[i]), 2)]
            conf_std_skf[i] = [round(stdev(config5[i]), 2), round(stdev(config6[i]), 2), round(stdev(config4[i]), 2)]

        #sort = sorted(conf_avg_skf, key=conf_avg_skf.get)
        x = [i for i in sort]

        plt.plot(x, [conf_avg_skf[i][2] for i in sort], color='brown', label='K = 3\nmax: '
                                                                + str(max([conf_avg_skf[i][2] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_skf[i][2] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_skf[i][2] for i in sort]) - np.array([conf_std_skf[i][2] for i in sort])/10,
                         np.array([conf_avg_skf[i][2] for i in sort]) + (np.array([conf_std_skf[i][2] for i in sort]))/10,
                         alpha=0.2, color='brown')

        plt.plot(x, [conf_avg_skf[i][0] for i in sort], color='olivedrab', label='K = 7\nmax: '
                                                                + str(max([conf_avg_skf[i][0] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_skf[i][0] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_skf[i][0] for i in sort]) - np.array([conf_std_skf[i][0] for i in sort])/10,
                         np.array([conf_avg_skf[i][0] for i in sort]) + (np.array([conf_std_skf[i][0] for i in sort]))/10,
                         alpha=0.2, color='olivedrab')

        plt.plot(x, [conf_avg_skf[i][1] for i in sort], color='royalblue', label='K = 10\nmax: '
                                                                + str(max([conf_avg_skf[i][1] for i in sort]))
                                                                + "\nmin: "
                                                                + str(min([conf_avg_skf[i][1] for i in sort])))
        plt.fill_between(x, np.array([conf_avg_skf[i][1] for i in sort]) - np.array([conf_std_skf[i][1] for i in sort])/10,
                         np.array([conf_avg_skf[i][1] for i in sort]) + (np.array([conf_std_skf[i][1] for i in sort]))/10,
                         alpha=0.2, color='royalblue')

        plt.ylabel('Test Accuracy')
        plt.xlabel('Classifier Configuration')
        plt.legend(fontsize=12)
        plt.ylim([0.45, 0.95])
        plt.grid()
        index = np.linspace(0, len(sort) - 1, 8, endpoint=True)
        plt.xticks([sort[ceil(i)] for i in index], rotation=10, fontsize=10)
        ax = plt.gca()
        ax.tick_params(width=2)
        plt.title("Stratified K Fold: " + classifier + " Classifier\n" + "Configuration: " + classifier_conf[classifier], fontweight="bold", fontsize=14)
        fig = plt.gcf()
        fig.set_size_inches((16, 9), forward=False)
        fig.savefig("./plot/" + dataset + "/single/skf/" + classifier + '.png', dpi=500)
        # plt.show()
        plt.clf()










