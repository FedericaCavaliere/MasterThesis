import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np
from math import ceil
import os
import glob
from numpy import mean


def find_genome(lines, i):

    for line in lines[i+1:]:
        if line.split(":")[0] == 'IndLenGenome':
            return int(line.split(":")[1])


def main(fold_name, n_gen_max):
    result = {}

    for filename in glob.glob(os.path.join("./" + fold_name + "/", '*.txt')):
        #print(filename)
        with open(filename, 'r') as fp:
            lines = fp.readlines()

        if list(lines[0])[8] == 'M':
            if list(lines[1])[3] == 'F':
                if "mf" not in result:
                    result["mf"] = {}
                seed = int(lines[2].split(":")[1])
                if seed not in result["mf"]:
                    result["mf"][seed] = {}
                k = int(lines[3].split(":")[1])
                if k not in result["mf"][seed]:
                    result["mf"][seed][k] = {}

                new_lines = lines[4:]
                for i in range(len(new_lines)):
                    if new_lines[i].split(":")[0] == 'Gen' and not new_lines[i+1].split(":")[0] == 'last_save':
                        gen = int(new_lines[i].split(":")[1])
                        if gen not in result["mf"][seed][k]:
                            gen_ind = find_genome(new_lines, i)
                            result["mf"][seed][k][gen] = [(float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                            float(new_lines[i + 2].split(":")[1]),
                                                            int(new_lines[i + 3].split(":")[1]), gen_ind)]
                        else:
                            gen_ind = find_genome(new_lines, i)
                            result["mf"][seed][k][gen].append((float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                            float(new_lines[i + 2].split(":")[1]),
                                                            int(new_lines[i + 3].split(":")[1]), gen_ind))

                        print(filename, "mf", gen, float(new_lines[i + 2].split(":")[1]))
            else:
                    if list(lines[1])[3] == 'T':
                        if "mt" not in result:
                            result["mt"] = {}
                        seed = int(lines[2].split(":")[1])
                        if seed not in result["mt"]:
                            result["mt"][seed] = {}
                        k = int(lines[3].split(":")[1])
                        if k not in result["mt"][seed]:
                            result["mt"][seed][k] = {}

                        new_lines = lines[4:]
                        for i in range(len(new_lines)):
                            if new_lines[i].split(":")[0] == 'Gen'and not new_lines[i+1].split(":")[0] == 'last_save':
                                gen = int(new_lines[i].split(":")[1])
                                if gen not in result["mt"][seed][k]:
                                    gen_ind = find_genome(new_lines, i)
                                    result["mt"][seed][k][gen] = [(float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                                   float(new_lines[i + 2].split(":")[1]),
                                                                   int(new_lines[i + 3].split(":")[1]), gen_ind)]
                                else:
                                    gen_ind = find_genome(new_lines, i)
                                    result["mt"][seed][k][gen].append((float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                                       float(new_lines[i + 2].split(":")[1]),
                                                                       int(new_lines[i + 3].split(":")[1]), gen_ind))
        else:
            if list(lines[1])[3] == 'F':
                if "sf" not in result:
                    result["sf"] = {}
                seed = int(lines[2].split(":")[1])
                if seed not in result["sf"]:
                    result["sf"][seed] = {}
                k = int(lines[3].split(":")[1])
                if k not in result["sf"][seed]:
                    result["sf"][seed][k] = {}

                new_lines = lines[4:]
                for i in range(len(new_lines)):
                    if new_lines[i].split(":")[0] == 'Gen' and not new_lines[i+1].split(":")[0] == 'last_save':
                        gen = int(new_lines[i].split(":")[1])
                        if gen not in result["sf"][seed][k]:
                            gen_ind = find_genome(new_lines, i)
                            result["sf"][seed][k][gen] = [(float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                           float(new_lines[i + 2].split(":")[1]),
                                                           int(new_lines[i + 3].split(":")[1]), gen_ind)]
                        else:
                            gen_ind = find_genome(new_lines, i)
                            result["sf"][seed][k][gen].append((float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                               float(new_lines[i + 2].split(":")[1]),
                                                               int(new_lines[i + 3].split(":")[1]), gen_ind))

                        print(filename, "sf", gen, float(new_lines[i + 2].split(":")[1]))

            else:
                if list(lines[1])[3] == 'T':
                    if "st" not in result:
                        result["st"] = {}
                    seed = int(lines[2].split(":")[1])
                    if seed not in result["st"]:
                        result["st"][seed] = {}
                    k = int(lines[3].split(":")[1])
                    if k not in result["st"][seed]:
                        result["st"][seed][k] = {}

                    new_lines = lines[4:]
                    for i in range(len(new_lines)):
                        if new_lines[i].split(":")[0] == 'Gen' and not new_lines[i+1].split(":")[0] == 'last_save':
                            gen = int(new_lines[i].split(":")[1])
                            gen_ind = find_genome(new_lines,i)
                            if gen not in result["st"][seed][k]:
                                result["st"][seed][k][gen] = [(float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                               float(new_lines[i + 2].split(":")[1]),
                                                               int(new_lines[i + 3].split(":")[1]), gen_ind)]
                            else:
                                gen_ind = find_genome(new_lines, i)
                                result["st"][seed][k][gen].append((float(new_lines[i + 1].split(":")[1]) - 0.02,
                                                                   float(new_lines[i + 2].split(":")[1]),
                                                                   int(new_lines[i + 3].split(":")[1]), gen_ind))
    for data in result:
        #print(data)
        tmp_avg = {}
        tmp_std = {}
        tmp_seed_avg = {}
        tmp_seed_std = {}
        tmp_k_avg = {}
        tmp_k_std = {}

        for seed in result[data]:
            #print(seed)
            if seed not in tmp_seed_avg:
                tmp_seed_avg[seed] = {}
                tmp_seed_std[seed] = {}

            for k in result[data][seed]:
                # print(k)
                if k not in tmp_k_avg:
                    tmp_k_avg[k] = {}
                    tmp_k_std[k] = {}

                last = result[data][seed][k][1]
                for gen in range(1, n_gen_max + 1):
                    if gen not in result[data][seed][k]:
                        result[data][seed][k][gen] = last
                    else:
                        last = result[data][seed][k][gen]

                for gen in result[data][seed][k]:
                    #print(gen)
                    result[data][seed][k][gen] = tuple(map(mean, zip(*result[data][seed][k][gen])))

                    if gen not in tmp_avg:
                        tmp_avg[gen] = [result[data][seed][k][gen]]
                    else:
                        tmp_avg[gen].append(result[data][seed][k][gen])

                    if gen not in tmp_seed_avg[seed]:
                        tmp_seed_avg[seed][gen] = [result[data][seed][k][gen]]
                    else:
                        tmp_seed_avg[seed][gen].append(result[data][seed][k][gen])

                    if gen not in tmp_k_avg[k]:
                        tmp_k_avg[k][gen] = [result[data][seed][k][gen]]
                    else:
                        tmp_k_avg[k][gen].append(result[data][seed][k][gen])

        for k in tmp_k_avg:
            for gen in tmp_k_avg[k]:
                if len(tmp_k_avg[k][gen]) > 1:
                    tmp_k_std[k][gen] = tuple(map(stdev, zip(*tmp_k_avg[k][gen])))
                else:
                    tmp_k_std[k][gen] = (0.0, 0.0, 0.0, 0.0)
                tmp_k_avg[k][gen] = tuple(map(mean, zip(*tmp_k_avg[k][gen])))

        for seed in tmp_seed_avg:
            for gen in tmp_seed_avg[seed]:
                if len(tmp_seed_avg[seed][gen]) > 1:
                    tmp_seed_std[seed][gen] = tuple(map(stdev, zip(*tmp_seed_avg[seed][gen])))
                else:
                    tmp_seed_std[seed][gen] = (0.0, 0.0, 0.0, 0.0)
                tmp_seed_avg[seed][gen] = tuple(map(mean, zip(*tmp_seed_avg[seed][gen])))

        for gen in tmp_avg:
            if len(tmp_avg[gen]) > 1:
                tmp_std[gen] = tuple(map(stdev, zip(*tmp_avg[gen])))
            else:
                tmp_std[gen] = (0.0, 0.0, 0.0, 0.0)
            tmp_avg[gen] = tuple(map(mean, zip(*tmp_avg[gen])))

        #print(data)
        for k in tmp_k_avg:
            #print(k)
            x = sorted(tmp_k_avg[k])
            y_avg = [tmp_k_avg[k][i][0] for i in x]
            y_std = [tmp_k_std[k][i][0] for i in x]
            #print(min(y_avg), max(y_avg), y_avg[0], y_avg[n_gen_max - 1])
            #print(min(y_std), max(y_std), y_std[0], y_std[n_gen_max - 1])

            plt.ylabel('Test Accuracy')
            plt.xlabel('Generations')

            s = ""
            if data == 'mt':
                s = "Meander Dataset with Features Selection "
            if data == 'sf':
                s = "Spiral Dataset"
            if data == 'st':
                s = "Spiral Dataset with Features Selection "
            if data == 'mf':
                s = "Meander Dataset"

            plt.title('Grammatical Evolution: ' + s, fontweight="bold", fontsize=14)
            plt.plot(x, y_avg, label='K = ' + str(k))
            plt.fill_between(x, np.array(y_avg) - np.array(y_std), np.array(y_avg) + np.array(y_std),
                             alpha=0.2, color='brown', label="Mean value ± Stdv value")
            plt.grid()
            plt.legend(fontsize=14, loc='lower right')
            #plt.show()
            plt.clf()

        x = sorted(tmp_avg.keys())
        y_avg = [tmp_avg[i][0] for i in x]
        y_std = [tmp_std[i][0] for i in x]
        plt.ylabel('Test Accuracy')
        plt.xlabel('Generations')

        #print(data)
        #print(min(y_avg), max(y_avg), y_avg[0], y_avg[n_gen_max-1])

        s = ""
        if data == 'mt':
            s = "Meander Dataset with Features Selection "
        if data == 'sf':
            s = "Spiral Dataset"
        if data == 'st':
            s = "Spiral Dataset with Features Selection "
        if data == 'mf':
            s = "Meander Dataset"

        plt.title('Grammatical Evolution: ' + s, fontweight="bold", fontsize=14)
        plt.plot(x, y_avg, label="Mean Accuracy")
        plt.fill_between(x, np.array(y_avg) - np.array(y_std), np.array(y_avg) + np.array(y_std),
                          alpha=0.2, color='brown', label="Mean value ± Stdv value")
        plt.legend(fontsize=16, loc='lower right')
        plt.grid()
        #plt.show()
        plt.clf()

        x = sorted(tmp_avg.keys())
        y_avg = [tmp_avg[i][2] for i in x]
        y_std = [tmp_std[i][2] for i in x]
        plt.ylabel('Genome Size')
        plt.xlabel('Generations')

        s = ""
        if data == 'mt':
            s = "Meander Dataset with Features Selection "
        if data == 'sf':
            s = "Spiral Dataset"
        if data == 'st':
            s = "Spiral Dataset with Features Selection "
        if data == 'mf':
            s = "Meander Dataset"

        plt.title('Grammatical Evolution: ' + s, fontweight="bold", fontsize=14)
        plt.plot(x, y_avg, label="Mean MaxGenomeLen")
        plt.fill_between(x, np.array(y_avg) - np.array(y_std), np.array(y_avg) + np.array(y_std),
                         alpha=0.2, color='brown', label="Stdv value ± Mean value")
        y_avg = [tmp_avg[i][3] for i in x]
        y_std = [tmp_std[i][3] for i in x]
        plt.plot(x, y_avg, label="Mean IndGenomeLen")
        plt.fill_between(x, np.array(y_avg) - np.array(y_std), np.array(y_avg) + np.array(y_std),
                         alpha=0.2, color='goldenrod', label="Stdv value ± Mean value")
        plt.legend(fontsize=10, loc='lower right')
        plt.grid()
        #plt.show()
        plt.clf()


name = ["adv_all", "basic_spiral", "basic_meander"]
gen_max = [2000, 500]

main(fold_name=name[0], n_gen_max=gen_max[0])
main(fold_name=name[1], n_gen_max=gen_max[1])
main(fold_name=name[2], n_gen_max=gen_max[1])
