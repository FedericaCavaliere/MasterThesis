import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from multiprocessing import Pool
from DT import main as main_DT
from NB import main as main_NB
from KNN import main as main_KNN
from RF import main as main_RF
from SVC import main as main_SVC
from AdaBoost import main as main_AB
from MLP import main as main_MLP


main_list = [main_DT, main_SVC, main_NB, main_KNN, main_RF, main_AB, main_MLP]
files_input = ["./dataset/spiral_full.txt", "./dataset/meander_full.txt"]
dir_output = ["./result/spiral/", "./result/meander/"]
dir_output_fs = ["./result/spiral_fs/", "./result/meander_fs/"]
features_selected = [0, 1, 2, 3, 4, 7]


def data(file_input, fselect=False):
    data = {'samples': [], 'labels': []}

    with open(file_input, "r") as fp:
        lines = fp.readlines()[1:]

    if not fselect:
        for line in lines:
            data['samples'].append([float(x) for x in line.split('\t')[:-1]])
            data['labels'].append(float(line[-2]))
    else:
        for line in lines:
            data['samples'].append([float(line.split('\t')[:-1][k]) for k in features_selected])
            data['labels'].append(float(line[-2]))
    return data


def exec(i, samples, labels, output):
    return main_list[i](samples, labels, output)


pool = Pool(None)
jobs = []

data_1 = data(files_input[0])
data_2 = data(files_input[1])
data_3 = data(files_input[0], True)
data_4 = data(files_input[1], True)

for i in range(len(main_list)):
    jobs.append(pool.apply_async(exec, (i, data_1['samples'], data_1['labels'], dir_output[0])))
    jobs.append(pool.apply_async(exec, (i, data_2['samples'], data_2['labels'], dir_output[1])))
    jobs.append(pool.apply_async(exec, (i, data_3['samples'], data_3['labels'], dir_output_fs[0])))
    jobs.append(pool.apply_async(exec, (i, data_4['samples'], data_4['labels'], dir_output_fs[1])))
for job in jobs:
    job.get(timeout=None)


