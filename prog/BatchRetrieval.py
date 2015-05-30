#! /usr/bin/env python2.7
from JunoCore import *
from multiprocessing import Process
import os, copy, time

## @file BatchRetrieval.py Submit a bunch of retrieval job and do it in parallel

## Retrieval using a single thread
# @param retrieval JunoRetrieval object 
# @param test_data_id Forward model filename 
# @param test_data_x observations with error
def single_thread_job(retrieval, 
        test_data_id, 
        test_data_x):
    filename = os.path.splitext(os.path.basename(test_data_id))[0]
    meta = get_metadata(test_data_id)
    t_true = meta[0]
    x_true = meta[1:]
    y_obs = test_data_x

    new_retrieval = copy.deepcopy(retrieval)
    folder = '../exp/retrieval_%s' % filename
    if not os.path.exists(folder): os.mkdir(folder)
    new_retrieval.model.header = folder + '/' + retrieval.model.header
    new_retrieval.report = folder + '/' + retrieval.report

    t_guess, x_guess = new_retrieval.guess(y_obs)
    t_result, x_result = new_retrieval.inverse(y_obs)
    with open(folder + '/result.txt', 'w') as report:
        report.write('%-15s' % t_true)
        for x in x_true:
            report.write('%-10.4f' % x)
        report.write('%-15s' % t_guess)
        for x in x_guess:
            report.write('%-10.4f' % x)
        for x in x_result:
            report.write('%-10.4f' % x)
        report.write('\n')

if __name__ == '__main__':
    ld_feature = { 
            'name'  : 'limb darkening',
            'weight': 1.0,
            'error' : 0.1E-2
            }

    bt_feature = {
            'name'  : 'brightness temperature',
            'weight': 0.0,
            'error' : 3.E-2
            }

    model = JunoAtmosphere(
            executable = '../bin/juno.ex', 
            input = '../bin/juno.in', 
            header = 'fwd-', 
            tailer = '.h5'
            )

    retrieval = JunoRetrieval(
            model = model, 
            features = [ld_feature, bt_feature],
            classifier = RidgeClassifier(alpha = 8.E5),
            guessor = DecisionTreeRegressor(min_samples_leaf = 4),
            report = 'status.txt'
            )

    retrieval.train()

    #test_data_x = genfromtxt('random-limbbrightness.txt')[:, 2:]
    #test_data_id = genfromtxt('random-limbbrightness.txt', usecols = (0,), dtype = str)
    test_data_x = genfromtxt('grid-limbbrightness.txt')[:, 2:]
    test_data_id = genfromtxt('grid-limbbrightness.txt', usecols = (0,), dtype = str)
    
    #test_start, test_end, n_threads = 0, len(test_data_id), 4
    # for fusi
    test_start, test_end, n_threads = 0, 100, 1
    # for ganesha
    #test_start, test_end, n_threads = 100, 300, 2
    # for aruba
    #test_start, test_end, n_threads = 300, 500, 1

    cur =  test_start + n_threads

    batch = [Process(target = single_thread_job, args = (retrieval, test_data_id[id], test_data_x[id])) for id in range(test_start, cur)]
    for p in batch: p.start()
    while cur < test_end:
        for i in range(n_threads):
            if not batch[i].is_alive():
                batch[i] = Process(target = single_thread_job, args = (retrieval, test_data_id[cur], test_data_x[cur]))
                batch[i].start()
                cur += 1
        time.sleep(10)
