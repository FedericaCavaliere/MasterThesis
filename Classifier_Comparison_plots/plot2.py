import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np
import matplotlib.cm as cm
from math import ceil

classifiers = ["DT", "KNN", "SVM", "AB", "RF", "MLP"]
datasets = ["spiral", "meander"]
classifier_conf = {"DT": "[max_depth]_[min_samples_split]_[min_samples_leaf]",
                   "NB": "[var_smoothing]", "KNN": "[neighbor]_[distance]", "SVM": "[kernel]_[gamma]_[C]_[degree]",
                   "AB": "[learning_rate]_[n_estimator]_[algorithm]",
                   "RF": "[n_estimator]_[max_depth]_[min_samples_leaf]_[min_samples_split]_[max_feature]",
                   "MLP": "[activation]_[solver]_[learning_rate]_[hidden_layer_size]"}

classifier_par = {"DT": "maxDepth_minSamplesSplit_minSamplesLeaf",
                   "NB": "varSmoothing", "KNN": "neighbor_distance", "SVM": "kernel_gamma_C_degree",
                   "AB": "learningRate_nEstimator_algorithm",
                   "RF": "nEstimator_maxDepth_minSamplesLeaf_minSamplesSplit_maxFeature",
                   "MLP": "activation_solver_learningRate_hiddenLayerSize"}


def plot_config(conf_list, config, classifier):

    conf_par = {}
    param_list = classifier_par[classifier].split("_")

    for conf in conf_list:
        for k in config:
            for i in range(len(conf.split("_"))):
                cnt = 0
                for ind in range(len(conf.split("_"))):
                    if ind != i:
                        if k.split("_")[ind] == conf.split("_")[ind]:
                            cnt += 1

                if cnt == len(param_list) - 1:
                    par = param_list[i]
                    if par in conf_par:
                        conf_par[par] = conf_par[par] + config[k]
                    else:
                        conf_par[par] = config[k]

    conf_par_avg = {}
    conf_par_std = {}
    for i in conf_par:
        conf_par_avg[i] = round(mean(conf_par[i]), 2)
        conf_par_std[i] = round(stdev(conf_par[i]), 2)

    legend = ""
    for i in range(len(param_list)):
        legend += param_list[i] + ": " + str(conf_par_std[param_list[i]]) + "\n"

    plt.errorbar([i for i in param_list], [conf_par_avg[i] for i in param_list],
                 yerr=[conf_par_std[i] for i in param_list], ecolor='brown', fmt='o', label=legend)
    plt.grid()
    plt.title(classifier + " Classifier: Parameter Relevance", fontweight="bold", fontsize=14)
    plt.legend()
    plt.ylim([0.45, 0.95])
    plt.xticks(fontsize=8)
    plt.ylabel('Test accuracy')
    plt.xlabel('Classifier Parameter')
    #plt.show()
    plt.clf()


def var_par(par_val, config, classifier):
    param_list = classifier_par[classifier].split("_")

    par_var = {}

    if len(par_val) > 0:
        for k in config:
            print("Config: " + k + " Acc: " + str(config[k]))
            conf_list = k.split("_")
            for i in range(len(param_list)):
                valid_keys = []
                for v in par_val[param_list[i]]:
                    new_conf_list = conf_list.copy()
                    new_conf_list[i] = v
                    new_k = "_".join(new_conf_list)
                    if new_k in config.keys():
                        print(new_k + " Acc: " + str(config[new_k]))
                        valid_keys.append(new_k)

                valid_keys = list(dict.fromkeys(valid_keys))

                if len(valid_keys) > 1:
                    if param_list[i] in par_var:
                        par_var[param_list[i]] = par_var[param_list[i]] + stdev([config[i] for i in valid_keys])
                    else:
                        par_var[param_list[i]] = stdev([config[i] for i in valid_keys])

    print(par_var)
    legend = ""

    for i in range(len(param_list)):
        legend += param_list[i] + ": " + str(round(par_var[param_list[i]],2)) + "\n"

    plt.scatter([i for i in param_list], [par_var[i] for i in param_list], label=legend)
    plt.grid()
    plt.legend()
    plt.title(classifier + " Classifier: Parameter Relevance", fontweight="bold", fontsize=14)
    plt.xticks(fontsize=8)
    plt.ylabel('Sum of variances')
    plt.xlabel('Classifier Parameter')
    plt.show()
    plt.clf()


def rank_in_cluster(cluster, param_list, occ_par_val, classifier):

    print(classifier)

    for k in cluster:
        occ = {}
        print()
        print("Cluster " + str(k))
        for conf in cluster[k]:
            conf_list = conf.split("_")
            for i in range(len(conf_list)):
                par = param_list[i]
                if par not in occ:
                    occ[par] = {}
                if str(conf_list[i]) not in occ[par]:
                    occ[par][str(conf_list[i])] = 1
                else:
                    occ[par][str(conf_list[i])] += 1

        for par in occ_par_val:
            print("Parameter " + par)
            for val in occ_par_val[par]:
                if val in occ[par]:
                    print("Occ of value " + str(val) + ": " + str(occ[par][val]) + "/" + str(occ_par_val[par][val]))
                else:
                    print("Occ of value " + str(val) + ": " + str(0) + "/" + str(occ_par_val[par][val]))

        print()


def main():

    for dataset in datasets:
        tot_cluster = {}
        tot_n_config = {}
        for classifier in classifiers:
            config = {}

            with open("./result/" + dataset + "/" + classifier + "_Split.txt", 'r') as fp:
                lines = fp.readlines()[1:]

            with open("./result/" + dataset + "/" + classifier + "_SKF.txt", 'r') as fp:
                for line in fp.readlines()[1:]:
                    lines.append(line)

            for line in lines:
                if classifier == "MLP":
                    key = line.split(",")[-2].replace("\n", "") + line.split(",")[-1].replace("\n", "")
                else:
                    key = line.split(",")[-1].replace("\n", "")

                newline = line.split(",")
                config.setdefault(key, []).append(float(newline[2]))

            conf_avg = {}
            conf_std = {}

            for i in config:
                conf_avg[i] = mean(config[i])
                conf_std[i] = stdev(config[i])

            sort = sorted(conf_avg, key=conf_avg.get)
            x = [i for i in range(len(sort))]

            plt.scatter(x, [conf_avg[i] for i in sort], label="Mean Accuracy")
            plt.fill_between(x, np.array([round(conf_avg[i], 2) for i in sort]) - (np.array([round(conf_std[i], 2) for i in sort])),
                             np.array([round(conf_avg[i], 2) for i in sort]) + (np.array([round(conf_std[i], 2) for i in sort])),
                             alpha=0.2, color='brown', label="Stdv value ± Mean value")

            plt.ylabel('Test Accuracy')
            plt.xlabel('Classifier Configuration')
            plt.legend(fontsize=12)
            plt.ylim([0.45, 0.95])
            plt.grid()
            plt.title(classifier + " Classifier", fontweight="bold", fontsize=14)
            fig = plt.gcf()
            fig.set_size_inches((16, 9), forward=False)
            fig.savefig("./plot/" + dataset + "/normal/" + classifier + ".png", dpi=500)
            #plt.show()
            plt.clf()

            cluster = {}
            for i in sort:
                near_key = round(float(conf_avg[i]) - 0.01, 2)

                if near_key in cluster:
                    cluster[near_key].append(i)
                else:
                    cluster.setdefault(round(conf_avg[i], 2), []).append(i)

            colors = iter(cm.rainbow(np.linspace(0, 1, len(cluster))))
            cnt = 0

            for i in cluster:
                plt.scatter([x for x in range(cnt, cnt + len(cluster[i]))], [i for _ in range(cnt, cnt + len(cluster[i]))],
                            color=next(colors))
                cnt += len(cluster[i])

            tot_cluster[classifier] = cluster
            tot_n_config[classifier] = len(config)

            plt.ylim([0.40, 0.80])
            plt.grid()
            plt.ylabel('Test Accuracy')
            plt.xlabel('Classifier Configuration')
            plt.yticks([i for i in cluster], fontsize=11)
            plt.title(classifier + " Classifier: Clustering", fontweight="bold", fontsize=14)
            # fig = plt.gcf()
            # fig.set_size_inches((16, 9), forward=False)
            # fig.savefig("./plot/" + dataset + "/cluster/" + classifier + ".png", dpi=500)
            #plt.show()
            plt.clf()

            param_list = classifier_par[classifier].split("_")
            clust_conf = []

            for k in cluster:
                conf = ""

                for ind in range(len(param_list)):
                    occ = []
                    for v in cluster[k]:
                        occ.append(v.split("_")[ind])
                    occ.sort()
                    conf += max(occ, key=occ.count)

                    flag = False
                    for i in config.keys():
                        if conf in i:
                            flag = True
                    if flag:
                        if ind != len(param_list) - 1:
                            conf += "_"
                    else:
                        occ.remove(max(occ, key=occ.count))
                        conf += max(occ, key=occ.count)

                if len(conf.split("_")) == len(param_list):
                    clust_conf.append(conf)

            for _ in clust_conf:
                 plot_config([_], config, classifier)

            par_val = {}
            occ_par_val = {}

            for k in conf_avg:
                conf_list = k.split("_")
                for i in range(len(conf_list)):
                    par = param_list[i]
                    if par not in par_val:
                        par_val[par] = []
                        occ_par_val[par] = {}
                    if str(conf_list[i]) not in par_val[par]:
                        par_val[par].append(str(conf_list[i]))
                        occ_par_val[par][str(conf_list[i])] = 1
                    else:
                        occ_par_val[par][str(conf_list[i])] += 1

            for i in par_val:
                par_val[i].sort()

            #print(dataset)
            #rank_in_cluster(cluster, param_list, occ_par_val, classifier)
            #var_par(par_val, conf_avg, classifier)

        colors = iter(cm.rainbow(np.linspace(0, 1, len(tot_cluster))))

        tmp = []
        for i in range(len(classifiers)):
            for k in tot_cluster[classifiers[i]]:
                if k not in tmp:
                    tmp.append(k)
        tmp.sort(reverse=True)

        bins = []
        for i in tmp:
            near = round(i + 0.01, 2)
            if near not in bins:
                bins.append(i)

        temp2 = {}
        for c in classifiers:
            temp2[c] = []
            for k in tmp:
                if k in tot_cluster[c]:
                    perc = ceil((len(tot_cluster[c][k])/tot_n_config[c])*100)
                    if k in bins:
                        for i in range(perc):
                            temp2[c].append(k)
                    else:
                        near = round(k + 0.01, 2)
                        for i in range(perc):
                            temp2[c].append(near)

        bins.sort()
        temp3 = {}
        width = 0.002
        shift = [-3, -2, -1, 0, 1, 2]

        v = [0.47, 0.49,0.51,0.53,0.55,0.57,0.59,0.62,0.64,0.66,0.68,0.70,0.72]
        for i in temp2:
            print(i)
            for k in v:
                print(k, temp2[i].count(k))

        for i in range(len(classifiers)):
            temp3[classifiers[i]] = [round(x-shift[i]*width - width, 4) for x in temp2[classifiers[i]]]

        print(temp3)

        for i in range(len(classifiers)):
            cond = [round(x-shift[i]*width - width, 4) for x in bins]
            plt.hist(list(temp3[classifiers[i]]), cond + [max(cond) + 0.01], width=width, color=next(colors),
                     label=classifiers[i])
        plt.legend()
        plt.grid()
        plt.xticks(bins)
        plt.xlabel('Test Accuracy')
        plt.ylabel('Perc. of Configurations')
        plt.title('Histogram of Accuracy Cluster', fontweight="bold", fontsize=14)
        # fig = plt.gcf()
        # fig.set_size_inches((16, 9), forward=False)
        # fig.savefig("./plot/" + dataset + "/hist.png", dpi=500)
        #plt.show()
        plt.clf()


main()

