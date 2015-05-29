#! /usr/bin/env python2.7
## @file SimulateScan.py File contains SimulateScan

## @package SimulateScan Perform a simulated pole-to-pole scan with
# a global NH3 and H2O enrichment factor as well as a latitude dependent stretch parameter
from JunoCore import *
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeRegressor
import time

## Read inputfile
# @param filename Input file name
def read_input_file(filename):
    with open(filename, 'r') as file:
        file.readline() # Ammonia enrichment factor
        xNH3 = float(file.readline())
        file.readline() # Water enrichment factor
        xH2O = float(file.readline())
        file.readline() # Limb darkening error
        ld_error = float(file.readline())
        file.readline() # Brightness temperature error
        bt_error = float(file.readline())
        file.readline() # Latitude
        lat = amap(float, file.readline().split())
        file.readline() # Stretch parameter
        stretch = amap(float, file.readline().split())

        assert len(lat) == len(stretch), "Latitude and Stretch must have the same length"

    return xNH3, xH2O, ld_error, bt_error, lat, stretch

## Create a synthetic data file
# @param retrieval JunoRetrieval object  
# @param filename output file name
# @param param_list This is a unpacked tuple returned from read_input_file
def create_synthetic_data(retrieval, 
        filename, 
        *param_list):
    xNH3, xH2O, ld_error, bt_error, lat, stretch = param_list

    with open(filename, 'w') as file:
        file.write('# Number of latitudes\n')
        file.write('%d\n' % len(lat))
        for i in range(len(lat)):
            y_true, y_obs = retrieval.create_synthetic_data(x_true = [xNH3, xH2O, stretch[i]])
            file.write('# Latitude No. %d\n' % (i + 1,))
            file.write('%f\n' % lat[i])
            file.write('# Synthetic observations\n')
            for j in range(len(y_obs)):
                file.write('%12.6f' % y_obs[j])
            file.write('\n')

## Read synthetic observation from file
# @param filename Synthetic data file name
def read_synthetic_data(filename):
    lat, y_obs = [], []
    with open(filename, 'r') as file:
        file.readline() # Number of latitudes
        n_lats = int(file.readline())
        for i in range(n_lats):
            file.readline() # Latitude
            lat.append(float(file.readline()))
            file.readline() # Observation
            y_obs.append(amap(float, file.readline().split()))
    return lat, y_obs


## This class does Juno retrieval for multiple observations
#
# JunoRetrieval only performs retrieval for a single observation.
# JunoMultiRetrieval extends the functionality of JunoRetrieval by allowing combining multiple observations
# so as to reduce the retrieval error.
class JunoMultiRetrieval(JunoRetrieval):
    # Base class constructor will be inherited

    ## Override the loss function in JunoRetrieval
    def loss_function(self, 
            params, 
            y_obs, 
            output = 'f'):
        if output == 'f':
            self.model.forward(parameter_to_list(params), 'f')
        y, fi, weight = self.read_feature(output)
        return array([weight * (y - y_obs[i]) for i in range(len(y_obs))]).flatten()

    ## Override write_iter_report in JunoRetrieval
    def write_iter_report(self, params, iter, res, *args, **kwargs):
        with open(self.report, 'a') as file:
            rmsd = sqrt(sum(res**2 / len(res)))
            file.write('Step #%i: Root-mean-square deviation= %10.2E\n' % (iter, rmsd))
            file.write(80 * '-' + '\n')
            file.write('Residual:\n')
            for i in range(len(res)):
                file.write('%12.4E' % res[i])
                if (i + 1) % 5 == 0: file.write('\n')
            file.write('\n')
            for key, info in params.items():
                file.write('%s = %12.4E\n' % (key, info.value))
            file.write(80 * '-' + '\n')

    ## Override inverse in JunoRetrieval
    def inverse(self, y_obs, sat_id, sat_guess, stretch_id, stretch_guess, lat):
        with open(self.report, 'w') as file:
            file.write('** Juno Retrieval Begin **\n')
            file.write('Saturation model, initial guess = ')
            for i in range(len(sat_guess)): 
                file.write('%f ' % sat_guess[i])
            file.write('\n')

        y_sat_obs = [y_obs[i] for i in sat_id]
        y_stretch_obs = [y_obs[i] for i in stretch_id]

        p = Parameters()
        p.add('NH3', value = sat_guess[0], min = 0.1, max = 20.)
        p.add('H2O', value = sat_guess[1], min = 0.1, max = 20.)
        p.add('Stretch', value = 1., vary = False)
        result = minimize(self.loss_function, p, iter_cb = self.write_iter_report,
                col_deriv = True, Dfun = self.jacobian, args = (y_sat_obs, 'f'),
                xtol = 1.E-4, ftol = 1.E-8, maxfev = 100)

        fNH3 = p['NH3'].value
        fH2O = p['H2O'].value

        fstretch = []
        with open(self.report, 'a') as file:
            file.write(fit_report(p))
            file.write('\nStretch model, initial guess =\n')
            for i in range(len(stretch_guess)):
                file.write('%12.4f ' % stretch_guess[i])
                if (i + 1) % 5 == 0: file.write('\n')
            file.write('\n')

        for i in range(len(y_stretch_obs)):
            with open(self.report, 'a') as file:
                file.write('\nLatitude No. %d: %8.4f\n' % (stretch_id[i] + 1, lat[stretch_id[i]]))
            p = Parameters()
            p.add('NH3', value = fNH3, vary = False)
            p.add('H2O', value = fH2O, vary = False)
            p.add('Stretch', value = stretch_guess[i], vary = True)
            result = minimize(self.loss_function, p, iter_cb = self.write_iter_report,
                    col_deriv = True, Dfun = self.jacobian, args = ([y_stretch_obs[i]], 'f'),
                    xtol = 1.E-4, ftol = 1.E-8, maxfev = 100)
            with open(self.report, 'a') as file:
                file.write(fit_report(p))
                file.write('\n')
            fstretch.append(p['Stretch'].value)

        with open(self.report, 'a') as file:
            file.write('\n')
            file.write('Fitted Parameters:\n')
            file.write('NH3 Enrichment Factor = %8.4f\n' % fNH3)
            file.write('H2O Enrichment Factor = %8.4f\n' % fH2O)
            file.write('%-12s%20s\n' % ('Latitude', 'Stretch Factor'))
            for i in range(len(y_obs)):
                if i in sat_id:
                    file.write('%-12.4f%20.4f\n' % (lat[i], 1.))
                else:
                    file.write('%-12.4f%20.4f\n' % (lat[i], fstretch[stretch_id.index(i)]))
            file.write('** Juno Retrieval End **\n')
            

if __name__ == '__main__':
    time_start = time.time()

    model = JunoAtmosphere(
            executable = '../z.JAMRT.Apr15/bin/juno.ex', 
            input = '../z.JAMRT.Apr15/bin/juno.in', 
            header = '../tmp/fwd-', 
            tailer = '.h5'
            )
    
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

    retrieval = JunoRetrieval(
            model = model, 
            features = [ld_feature, bt_feature],
            classifier = RidgeClassifier(alpha = 8.E5),
            guessor = DecisionTreeRegressor(min_samples_leaf = 4),
            report = '../tmp/status.txt'
            )

    retrieval.train()
    retrieval.insample_validation()
    retrieval.outsample_validation()

    #input = read_input_file('SimulateScan.inp')
    #create_synthetic_data(retrieval, 'SimulateScan.dat', *input)
    lat, y_obs = read_synthetic_data('SimulateScan.dat')

    sat_id, sat_guess = [], array([0., 0.])
    stretch_id, stretch_guess = [], []
    for i in range(len(lat)):
        guess = retrieval.guess(y_obs[i])
        print guess
        if guess[0] == 'sat':
            sat_id.append(i)
            sat_guess += guess[1]
        else:
            stretch_id.append(i)
            stretch_guess.append(guess[1][2])
    sat_guess = sat_guess / len(sat_id)

    mret = JunoMultiRetrieval(
            model = model,
            features = [ld_feature, bt_feature],
            report = '../tmp/report.txt'
            )
    mret.inverse(y_obs, sat_id, sat_guess, stretch_id, stretch_guess, lat)

    time_end = time.time()
    with open('../tmp/report.txt', 'a') as file:
        file.write('Elapsed time = %f s\n' % (time_end - time_start,))
