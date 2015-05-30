#! /usr/bin/env python2.7
from pylab import *
from true_vs_retrieval import read_monte_carlo_results

## @file misclassify.py Plot misclassified cases in a 3-d space (projected into 3 surfaces)

if __name__ == '__main__':
    filename = '../data/monte_carlo_results.txt'

    sat_sat_true, sat_sat_guess, sat_sat_result, \
    sat_str_true, sat_str_guess, sat_str_result, \
    str_sat_true, str_sat_guess, str_sat_result, \
    str_str_true, str_str_guess, str_str_result = read_monte_carlo_results(filename)

    fig, axs = subplots(nrows = 2, ncols = 2, figsize = (12, 10))
    fig.subplots_adjust(wspace = 0.25, hspace = 0.35)
    
    # sat classification, NH3 and H2O
    ax = axs[0, 0]
    ax.scatter(sat_sat_true[:, 0], sat_sat_true[:, 1], alpha = 0.5, c = 'b', linewidth = 0,
            s = 1000 * abs(sat_sat_result[:, 1] - sat_sat_true[:, 1]) / sat_sat_true[:, 1])
    ax.scatter(sat_str_true[:, 0], sat_str_true[:, 1], alpha = 0.5, c = 'r', linewidth = 0, s = 100)
    accuracy = 100. * len(sat_sat_true) / (len(sat_sat_true) + len(sat_str_true))
    ax.set_title('%4.2f' % accuracy + ' %', fontsize = 20)
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])
    ax.set_xlabel('NH3')
    ax.set_ylabel('H2O')

    # stretch classification, NH3 and H2O 
    ax = axs[0, 1]
    ax.scatter(str_str_true[:, 0], str_str_true[:, 1], alpha = 0.5, c = 'b', linewidth = 0,
            s = 1000 * abs(str_str_result[:, 2] - str_str_true[:, 2]) / str_str_true[:, 2])
    ax.scatter(str_sat_true[:, 0], str_sat_true[:, 1], alpha = 0.5, c = 'r', linewidth = 0, s = 100)
    accuracy = 100. * len(str_str_true) / (len(str_str_true) + len(str_sat_true))
    ax.set_title('%4.2f' % accuracy + ' %', fontsize = 20)
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])
    ax.set_xlabel('NH3')
    ax.set_ylabel('H2O')

    # stretch classification, NH3 and Stretch
    ax = axs[1, 0]
    ax.scatter(str_str_true[:, 0], str_str_true[:, 2], alpha = 0.5, c = 'b', linewidth = 0,
            s = 1000 * abs(str_str_result[:, 2] - str_str_true[:, 2]) / str_str_true[:, 2])
    ax.scatter(str_sat_true[:, 0], str_sat_true[:, 2], alpha = 0.5, c = 'r', linewidth = 0, s = 100)
    ax.set_title('%4.2f' % accuracy + ' %', fontsize = 20)
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 5])
    ax.set_xlabel('NH3')
    ax.set_ylabel('Stretch')

    # stretch classification, H2O and Stretch
    ax = axs[1, 1]
    ax.scatter(str_str_true[:, 1], str_str_true[:, 2], alpha = 0.5, c = 'b', linewidth = 0,
            s = 1000 * abs(str_str_result[:, 2] - str_str_true[:, 2]) / str_str_true[:, 2])
    ax.scatter(str_sat_true[:, 1], str_sat_true[:, 2], alpha = 0.5, c = 'r', linewidth = 0, s = 100)
    ax.set_title('%4.2f' % accuracy + ' %', fontsize = 20)
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 5])
    ax.set_xlabel('H2O')
    ax.set_ylabel('Stretch')

    #show()
    savefig('../figure/misclassify.png', bbox_inches = 'tight')
