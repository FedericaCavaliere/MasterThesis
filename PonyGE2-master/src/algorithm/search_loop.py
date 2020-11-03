from multiprocessing import Pool
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from utilities.stats import trackers
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init
from sklearn.model_selection import StratifiedKFold
from utilities.fitness.get_data import get_data
import numpy as np
import time
from os import path


def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    sort = sorted(individuals, key=lambda x: len(x.genome), reverse=True)
    print("Max init genome length: ", len(sort[0].genome))

    x_train, y_train, x_test, y_test = get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
    params['x_train'] = x_train
    params['x_test'] = x_test
    params['y_train'] = y_train
    params['y_test'] = y_test

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # for ind in individuals:
    #     print(ind.fitness)

    # Generate statistics for run so far
    get_stats(individuals)

    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation
        # New generation
        individuals = params['STEP'](individuals)
        best = max(individuals)
        best.tree.print_tree()
        print(best)
        # print("********************************", generation, "********************************")
        # for ind in individuals:
        #     print(ind)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals


def new_search_loop(last_save=None, fselect=False, checkpoint=False):
    # last_save = [last_size, last_k]

    params['RANDOM_SEED'] = 1234
    pop_size = params['POPULATION_SIZE']
    n_gen = params['GENERATIONS']

    step_gen = 50
    step_genome = 15

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    threshold = 0.9
    individuals = None

    data = {'samples': [], 'labels': []}

    with open(path.abspath("../datasets/" + params['DATASET_FULL']).replace("src/", ""), "r") as fp:
        lines = fp.readlines()[1:]

        for line in lines:
            data['samples'].append([float(x) for x in line.split()[:-1]])
            data['labels'].append(int(line[-2]))

    cnt_trial = 0
    for size in [10, 7, 3]:
        if not checkpoint or (checkpoint and size >= last_save[0]):
            skf = StratifiedKFold(n_splits=size, random_state=1234)
            X = np.asarray(data['samples'])
            Y = np.asarray(data['labels'])

            for k in range(3):
                if not checkpoint or (checkpoint and k >= last_save[1]):
                    params['RANDOM_SEED'] = 1234 + k
                    filename = time.strftime("%Y_%m_%d-%H_%M_%S")
                    with open("result_big_run/" + filename + ".txt", "a") as fp:
                        fp.write("Dataset:" + str(params['DATASET_FULL']) + "\nFS:" + str(fselect) +
                                        "\nSeed:" + str(params['RANDOM_SEED']) + "\nSize:" + str(size))

                    cnt_for = 0
                    cnt_trial += 1
                    print(str(cnt_trial) + "/9")

                    for train_index, test_index in skf.split(data['samples'], data['labels']):
                        cnt_for += 1
                        print(str(cnt_for) + "/" + str(size))

                        x_train, x_test = X[train_index], X[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]

                        params['x_train'] = x_train
                        params['x_test'] = x_test
                        params['y_train'] = y_train
                        params['y_test'] = y_test

                        # Initialise population
                        individuals = initialisation(pop_size)

                        # Evaluate initial population
                        individuals = evaluate_fitness(individuals)

                        # Generate statistics for run so far
                        get_stats(individuals)
                        cnt = 0
                        old_best_fitness = -1
                        cnt_stg = 0
                        params['MAX_GENOME_LENGTH'] = 400

                        # Traditional GE
                        while old_best_fitness < threshold and cnt < 500 and cnt_stg < 250:
                            cnt += 1
                            stats['gen'] = cnt
                            print("****** Gen ", cnt, " ******")

                            if cnt > n_gen:
                                params['GENERATIONS'] = cnt

                            best = max(individuals)
                            test_fitness = params['FITNESS_FUNCTION'](best, x=params['x_test'], y=params['y_test'])
                            #print(best.fitness, test_fitness)

                            if best.fitness > old_best_fitness:
                                cnt_stg = 0
                                #print("BestTrain: ", best.fitness)
                                #print("Best Test: ", test_fitness)
                                old_best_fitness = best.fitness

                                with open("result_big_run/" + filename + ".txt", "a") as fp:
                                    fp.write("\n\nGen:" + str(cnt) + "\nTrainFitness:" + str(best.fitness) +
                                             "\nTestFitness:" + str(test_fitness) + "\nMaxGenLen:" + str(params['MAX_GENOME_LENGTH']) +
                                             "\n" + str(best) + "\nIndLenGenome:" + str(len(best.genome)) + "\nIndGenome:" + str(best.genome) +
                                             "\nlast_save: " + str(size) + str(k))

                            else:
                                cnt_stg += 1

                                if params['MAX_GENOME_LENGTH']:
                                    if cnt_stg % step_gen == 0:
                                        params['MAX_GENOME_LENGTH'] += step_genome
                                        #print("MaxGenLen: ", params['MAX_GENOME_LENGTH'])
                                        #print("NGenStagn: ", cnt_stg)
                                else:
                                    if cnt_stg % step_gen == 0:
                                        params['MAX_GENOME_LENGTH'] = step_genome
                                        #print("MaxGenLen: ", params['MAX_GENOME_LENGTH'])
                                        #print("NGenStagn: ", cnt_stg)

                            # New generation
                            individuals = params['STEP'](individuals)

                        print("Exit:" + str(old_best_fitness >= threshold) + str(params['GENERATIONS'] >= 500) +
                              str(cnt_stg >= 250))
                        with open("result_big_run/" + filename + ".txt", "a") as fp:
                            fp.write("\n\nGen:" + str(cnt) + "\nlast_save: " + str(size) + str(k) +
                                     "\nExit:" + str(old_best_fitness >= threshold) + str(params['GENERATIONS'] >= 500)
                                     + str(cnt_stg >= 250))


    #print("BEST", best.fitness)
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """
    
    individuals = trackers.state_individuals
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)
    
    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        
        # New generation
        individuals = params['STEP'](individuals)
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    return individuals

