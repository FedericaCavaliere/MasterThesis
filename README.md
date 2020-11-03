# MasterThesis
This repository contains 4 project folders:
1) "Classifier_Comparison" for the benchmarking work in which the performances (as the hyperparameters vary) of 6 machine learning techniques (SVM, Naive Bayes, MLP, AdaBoost, Random Forest, Decision Tree) were compared on the data sets meanders and spirals HandPD;
2) "Classifier_Comparison_plots" for the graphs of point 1);
3) "PonyGE2-master" for the performance analysis of the Grammatical Evolution technique with the homonymous Python library (PonyGE2) on the HandPD dataset, designed two grammars (basic grammar, advanced grammar);
4) "GE_plots" for the graphs of point 3).

For project 3),

- the executable source file is recoverable for testing at the following path: PonyGE2-master / src / big_run.py

For a correct execution,

- set as execution parameters the name of the evolution classification file chosen (--parameters _namefile_), which can be viewed in: PonyGE2-master / parameters /, a classification file for the meadri (classification_meander.txt) and one for the spirals (classification_spiral.txt)
- the grammar used is specified in the classification file, which can be viewed in: PonyGE2-master / parameters / grammars, a file for basic grammar (test_basic.pybnf) and one for advanced grammar (test_adv.pybnf)
- the function for calculating and evaluating the fitness of each individual can be viewed at: PonyGE2-master / src / fitness / supervised_learning / adf_supervised_learning.py
