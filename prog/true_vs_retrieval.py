#! /usr/bin/env python2.7
import matplotlib
from pylab import *

## @file true_vs_retrieval.py Plot Monte Carlo retrieved results against true statevector

## Read a Monte Carlo results file
def read_monte_carlo_results(filename):
    with open(filename, 'r') as file:
        sat_sat_true, sat_sat_guess, sat_sat_result = [], [], []
        sat_str_true, sat_str_guess, sat_str_result = [], [], []
        str_sat_true, str_sat_guess, str_sat_result = [], [], []
        str_str_true, str_str_guess, str_str_result = [], [], []
        line = file.readline().split()
        while len(line) > 0:
            if line[0] == 'sat':
                true = [float(line[1]), float(line[2]), 1.]
                if line[3] == 'sat':
                    guess = [float(line[4]), float(line[5]), 1.]
                    result = [float(line[6]), float(line[7]), 1.]
                    sat_sat_true.append(true)
                    sat_sat_guess.append(guess)
                    sat_sat_result.append(result)
                elif line[3] == 'stretch':
                    guess = [float(line[4]), float(line[5]), float(line[6])]
                    result = [float(line[7]), float(line[8]), float(line[9])]
                    sat_str_true.append(true)
                    sat_str_guess.append(guess)
                    sat_str_result.append(result)
            elif line[0] == 'stretch':
                true = [float(line[1]), float(line[2]), float(line[3])]
                if line[4] == 'sat':
                    guess = [float(line[5]), float(line[6]), 1.]
                    result = [float(line[7]), float(line[8]), 1.]
                    str_sat_true.append(true)
                    str_sat_guess.append(guess)
                    str_sat_result.append(result)
                elif line[4] == 'stretch':
                    guess = [float(line[5]), float(line[6]), float(line[7])]
                    result = [float(line[8]), float(line[9]), float(line[10])]
                    str_str_true.append(true)
                    str_str_guess.append(guess)
                    str_str_result.append(result)
            line = file.readline().split()
    sat_sat_true, sat_sat_guess, sat_sat_result = array(sat_sat_true), array(sat_sat_guess), array(sat_sat_result)
    sat_str_true, sat_str_guess, sat_str_result = array(sat_str_true), array(sat_str_guess), array(sat_str_result)
    str_sat_true, str_sat_guess, str_sat_result = array(str_sat_true), array(str_sat_guess), array(str_sat_result)
    str_str_true, str_str_guess, str_str_result = array(str_str_true), array(str_str_guess), array(str_str_result)

    return  sat_sat_true, sat_sat_guess, sat_sat_result, \
            sat_str_true, sat_str_guess, sat_str_result, \
            str_sat_true, str_sat_guess, str_sat_result, \
            str_str_true, str_str_guess, str_str_result

if __name__ == '__main__':
    filename = '../data/monte_carlo_results.txt'

    sat_sat_true, sat_sat_guess, sat_sat_result, \
    sat_str_true, sat_str_guess, sat_str_result, \
    str_sat_true, str_sat_guess, str_sat_result, \
    str_str_true, str_str_guess, str_str_result = read_monte_carlo_results(filename)

    fig, axs = subplots(nrows = 2, ncols = 3, figsize = (16, 10), sharex = False)
    fig.subplots_adjust(wspace = 0.15, hspace = 0.12)
    # NH3 for sat
    ax = axs[0, 0]
    ax.plot(sat_sat_true[:, 0], sat_sat_true[:, 0], 'k')
    ax.plot(sat_sat_true[:, 0], sat_sat_guess[:, 0], 'bo', mec = 'b')
    ax.plot(sat_sat_true[:, 0], sat_sat_result[:, 0], 'ro', mec = 'r')
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])
    ax.set_ylabel('Retrieved value')

    # H2O for sat
    ax = axs[0, 1]
    ax.plot(sat_sat_true[:, 1], sat_sat_true[:, 1], 'k')
    ax.plot(sat_sat_true[:, 1], sat_sat_guess[:, 1], 'bo', mec = 'b')
    ax.plot(sat_sat_true[:, 1], sat_sat_result[:, 1], 'ro', mec = 'r')
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])

    # hide axis
    axs[0, 2].set_frame_on(False)
    axs[0, 2].get_xaxis().set_visible(False)
    axs[0, 2].get_yaxis().set_visible(False)

    # NH3 for str
    ax = axs[1, 0]
    ax.plot(str_str_true[:, 0], str_str_true[:, 0], 'k')
    ax.plot(str_str_true[:, 0], str_str_guess[:, 0], 'bo', mec = 'b')
    ax.plot(str_str_true[:, 0], str_str_result[:, 0], 'ro', mec = 'r')
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])
    ax.set_xlabel('True NH3')
    ax.set_ylabel('Retrieved value')

    # H2O for sat
    ax = axs[1, 1]
    ax.plot(str_str_true[:, 1], str_str_true[:, 1], 'k')
    ax.plot(str_str_true[:, 1], str_str_guess[:, 1], 'bo', mec = 'b')
    ax.plot(str_str_true[:, 1], str_str_result[:, 1], 'ro', mec = 'r')
    ax.set_xlim([1, 10])
    ax.set_ylim([1, 10])
    ax.set_xlabel('True H2O')

    # stretch for sat
    ax = axs[1, 2]
    ax.plot(str_str_true[:, 2], str_str_true[:, 2], 'k')
    ax.plot(str_str_true[:, 2], str_str_guess[:, 2], 'bo', mec = 'b')
    ax.plot(str_str_true[:, 2], str_str_result[:, 2], 'ro', mec = 'r')
    ax.set_xlim([1, 5])
    ax.set_ylim([1, 5])
    ax.set_xlabel('True Stretch')

    savefig('../true_vs_retrieval.png', bbox_inches = 'tight')
