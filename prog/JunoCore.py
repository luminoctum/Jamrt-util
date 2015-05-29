#! /usr/bin/env python2.7
## @file JunoCore.py File contains JunoCore

## @package JunoCore Module for handeling JAMRT output file, run a forward model,
# doing retrievals etc.
#
# This is a core module for handeling JAMRT program. It has two key classes:
# JunoAtmosphere and JunoRetrieval with several helper functions.
from pylab import *
from collections import Counter
from lmfit import minimize, Parameters, fit_report
from numpy.random import permutation
from sklearn.cross_validation import cross_val_score
import h5py, subprocess, os, operator, random, glob


## Read a parameter list, return a parameter string concatenated by '-'
# used to generate forward model output file name
# @param param Python list of parameters
def parameter_to_string(params):
    return reduce(operator.add, ['-'+str(x) for x in params[1:]], str(params[0]))

## Read output file name and return the parameters, inverse process of parameter_to_string
# @param filename Forward model file name
def get_metadata(filename):
    filename = os.path.splitext(os.path.basename(filename))[0]
    field = filename.split('-')[1:]
    for i in range(1, len(field)): field[i] = float(field[i])
    return field

## Change the Parameter object in lmfit to a python list
# @param params Lmfit Parameter() object
def parameter_to_list(params):
    return [params[key].value for key in params.keys()]

## Find the most common member in a list
# @param lst Python list
def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

## Create a grid statevector used for training a learning model
# @param model JunoAtmosphere object
# @param types 'sat' or 'stretch'
# @param xNH3 NH3 mixing ratio
# @param xH2O H2O mixing ratio
# @param folder folder containing the model outputs
# @param **kwargs a dictionary containers statevector other than NH3 or H2O
def create_grid_pool(model, 
        types = [], 
        xNH3 = [], 
        xH2O = [], 
        folder = '../data/juno_grid_pool/', 
        **kwargs):
    if not os.path.exists(folder): os.mkdir(folder)

    header = folder + model.header
    for model_type in types:
        if model_type == 'sat':
            model.header = header.replace('type', model_type)
            for a in xNH3:
                for b in xH2O:
                    param = [a, b, 1]
                    model.prepare(param, parameter_to_string(param[:-1]))

        if model_type == 'stretch':
            model.header = header.replace('type', model_type)
            for a in xNH3:
                for b in xH2O:
                    for c in kwargs['stretch']:
                        param = [a, b, c]
                        model.prepare(param, parameter_to_string(param))
        model.forward(threads = 16)

## Create a grid statevector used for training a learning model
# @param model JunoAtmosphere object
# @param types 'sat' or 'stretch'
# @param xNH3 NH3 mixing ratio
# @param xH2O H2O mixing ratio
# @param folder folder containing the model outputs
# @param **kwargs a dictionary containers statevector other than NH3 or H2O
def create_random_pool(model, 
        types = [], 
        num = 100, 
        xNH3 = (), 
        xH2O = (), 
        folder = '../data/juno_random_pool/', 
        **kwargs):
    if not os.path.exists(folder): os.mkdir(folder)

    header = folder + model.header
    for i in range(num):
        a = float('%.4f' % uniform(*xNH3))
        b = float('%.4f' % uniform(*xH2O))
        j = randint(0, len(types))
        model.header = folder + model.header + types[j] + '-'
        if types[j] == 'sat':
            param = [a, b, 1]
            model.header = header.replace('type', types[j])
            model.prepare(param, parameter_to_string(param[:-1]))

        if types[j] == 'stretch':
            c = float('%.4f' % uniform(*kwargs['stretch']))
            param = [a, b, c]
            model.header = header.replace('type', types[j])
            model.prepare(param, parameter_to_string(param))
    model.forward(threads = 16)

## JunoAtmosphere is responsible for read and run the JAMRT model
#
# The constructor of JunoAtmosphere defines the JAMRT executable file,
# model input file, the directory to put the output file (header)
# and the extension of the output file (tailer). If none of these
# are given, this class can be used as a class to read JAMRT outputs.
class JunoAtmosphere:

    ## Constructor
    # @param executable The JAMRT executable file.
    # @param input The JAMRT input file.
    # @param header The folder to put model output result.
    # @param tailer The extension of an output file.
    def __init__(self, 
            executable = None, 
            input = None, 
            header = '', 
            tailer = '.h5'):
        self.header = header
        self.tailer = tailer
        self.exe = executable
        self.input = input
        self.script = [self.exe, '--input', input, '--forward']
        self.model_list = []

    ## Run forward model
    # @param params Statevector to run the forward model. 
    # If this is unspecified, the forward model will be run in a parallel way.
    # @param output Forward model output file.
    # @param threads Number of threads to run the forward model.
    # 
    # The forward model can be run either in a serial way or a parallel way.
    def forward(self, 
            params = None, 
            output = '', 
            threads = 1):
        # parallel call
        if params == None:
            process_list = []
            for i in range(len(self.model_list) / threads):
                for j in range(threads): 
                    process_list.append(subprocess.Popen(self.model_list[j + i * threads]))
                for j in range(threads):
                    process_list[j + i * threads].communicate()
            i = len(self.model_list) / threads
            for j in range(i * threads, len(self.model_list)):
                process_list.append(subprocess.Popen(self.model_list[j]))
            for j in range(i * threads, len(self.model_list)):
                process_list[j].communicate()
            print 'Finished!'
            self.model_list = []
        # single call
        else:
            script = self.script + [str(x) for x in params]
            script += ['--output', self.header + output + self.tailer]
            #print script
            subprocess.call(script)

    ## Prepare to run forward model in a parallel way
    # @param params Statevector to run the forward model.
    # @param output Forward model output file.
    def prepare(self, 
            params, 
            output):
        script = self.script + [str(x) for x in params]
        script += ['--output', self.header + output + self.tailer]
        self.model_list.append(script)

    ## Read JAMRT output file
    # @param filename JAMRT output filename.
    # @param raw If this is true, filename is treated as the exact filename.
    # otherwise header and tailer will be attached to filename.
    def read(self, 
            filename, 
            raw = False):
        if raw:
            file = h5py.File(filename, 'r')
        else:
            file = h5py.File(self.header + filename + self.tailer, 'r')
        data = file['/output/vmr_H2O'][:]
        self.n_states = data.shape
        self._read_output_data('xH2O', file['/output/vmr_H2O'])
        self._read_output_data('xNH3', file['/output/vmr_NH3'])
        self._read_output_data('xH2S', file['/output/vmr_H2S'])
        self._read_output_data('pres', file['/output/pressure'])
        self._read_output_data('temp', file['/output/temperature'])
        self._read_output_data('bt', file['/output/brightness temperature'])
        self._read_input_data('freq', file['/input/frequencies'])
        self._read_input_data('angle', file['/input/angles'])
        file.close()

    ## Read input parameters
    # @param name Input parameter name.
    # @param h5data h5data variable.
    #
    # This function is supposed to be called by read and used internally
    def _read_input_data(self, 
            name, 
            h5data):
        setattr(self, name, h5data[:])

    ## Read output fields 
    # @param name Output field name.
    # @param h5data h5data variable.
    #
    # This function is supposed to be called by read and used internally
    def _read_output_data(self, 
            name, 
            h5data):
        if len(self.n_states) == 1:
            if prod(self.n_states) > 1:
                setattr(self, name, [])
                for i in range(self.n_states[0]):
                    getattr(self, name).append(h5data[:][i])
            else:
                setattr(self, name, h5data[:][0])
        if len(self.n_states) == 2:
            if prod(self.n_states) > 1:
                setattr(self, name, [])
                for i in range(self.n_states[0]):
                    for j in range(self.n_states[1]):
                        getattr(self, name).append(h5data[:][i, j])
            else:
                setattr(self, name, h5data[:][0, 0])
        if len(self.n_states) == 3:
            if prod(self.n_states) > 1:
                setattr(self, name, [])
                for i in range(self.n_states[0]):
                    for j in range(self.n_states[1]):
                        for k in range(self.n_states[2]):
                            getattr(self, name).append(h5data[:][i, j, k])
            else:
                setattr(self, name, h5data[:][0, 0, 0])
        if len(self.n_states) == 4:
            if prod(self.n_states) > 1:
                setattr(self, name, [])
                for i in range(self.n_states[0]):
                    for j in range(self.n_states[1]):
                        for k in range(self.n_states[2]):
                            for m in range(self.n_states[3]):
                                getattr(self, name).append(h5data[:][i, j, k, m])
            else:
                setattr(self, name, h5data[:][0, 0, 0, 0])


## JunoRetrieval is responsible for performing retrieval based on
# limb darkening and brightness temperature observations.
#  
# The constructor of JunoRetrieval takes the object of JunoAtmosphere,
# the feature vectors (python dictionary) to use, 
# the classifier to determine what kind of dynamic parameterization
# the gueesor to determine the approximate statevector given the dynamic parameterization
# a small number used to calculate jacobian
# and the filename to write the retrieval report

class JunoRetrieval:

    ## Constructor
    # @param model The object of class JunoAtmosphere, used to run the forward model
    # @param features This is a python list specifing the features to use for the retrieval. Each feature is a python dictionary
    # @param classifier Classifier is used to train and predict what kind of dynamic parameterization should be given the observations.
    # @param guessor Given a dynamic parameterization, guessor will try to "guess" the approximate value for the state vectors on a coarse grid.
    # @param eps This is a small number used to calculate the finite difference jacobian.
    # @param report This is the filename to write the report file.
    def __init__(self, 
            model = None, 
            features = [], 
            classifier = None, 
            guessor = None, 
            eps = 1.E-3, 
            report = 'status.txt'):
        self.model = model
        self.features = features
        self.n_features = len(features)
        self.feature_name = ''.join([x['name'].split(' ')[0] for x in features])
        self.classifier = classifier
        self.guessor = guessor
        self.eps = eps
        self.report = report

    ## Return a feature vector given the output file from a forward model.
    # @param output Forward model output file.
    # @param raw Flag to indicate whether filename is treat as exact or should append header and tailer to it.
    #
    # JunoRetrieval support for multi-feature retrieval, meaning that you can specify and combine any numbers of features. The features will be read seperately and combined into a flat array based on weights and errors.
    def read_feature(self, 
            output, 
            raw = False):
        feature_vector = array([])
        self.model.read(output, raw)

        feature_index, weight = [], []
        for id, feature in enumerate(self.features):
            if feature['name'] == 'limb darkening':
                for i in range(1, len(self.model.angle)):
                    data = self.model.bt[i, :] / self.model.bt[0, :]
                    feature_index.append([id, i - 1, len(data)])
                    feature_vector = hstack((feature_vector, data))
                    try: # if weight is a list
                        weight.append(list(feature['weight']))
                    except TypeError: # if weight is a scalar
                        weight.append([feature['weight']] *  len(data))

            if feature['name'] == 'brightness temperature':
                for i in range(len(self.model.angle)):
                    data = self.model.bt[i, :]
                    feature_index.append([id, i, len(data)])
                    feature_vector = hstack((feature_vector, data))
                    try: # if weight is a list
                        weight.append(list(feature['weight']))
                    except TypeError: # if weight is a scalar
                        weight.append([feature['weight']] *  len(data))

        # flatten weight array
        weight = array([item for sublist in weight for item in sublist])

        return feature_vector, feature_index, weight

    ## Return a synthetic observation given a statevector
    # @param x_true True statevector
    # @param output Forward model output file
    # @param raw Flag to indicate whether filename is treat as exact or should append header and tailer to it.
    def create_synthetic_data(self, 
            x_true = None, 
            output = None, 
            raw = False):
        if x_true != None:
            output = parameter_to_string(x_true)
            self.model.forward(x_true, output)
            y_true, feature_index, weight = self.read_feature(output, raw)
        elif output != None:
            y_true, feature_index, weight = self.read_feature(output, raw)
        else:
            raise ValueError('Either x_true or output should be specified')
        y_obs = zeros(len(y_true))

        cur = 0
        for fid in feature_index:
            for j in range(fid[2]):
                try: # if error is a list
                    y_obs[cur + j] = y_true[cur + j] * (1. + random.gauss(0, self.features[fid[0]]['error'][j]))
                except TypeError: # if error is a scalar
                    y_obs[cur + j] = y_true[cur + j] * (1. + random.gauss(0, self.features[fid[0]]['error']))
            cur += fid[2]

        return y_true, y_obs

    ## Create a feature file for training
    # @param folder Folder to contain the forward model output file
    # @param type Type can either be 'grid' or 'random'.
    def create_feature_file(self, 
            folder, 
            type = 'grid'):
        files = glob.glob(folder + '/*.h5')
        feature_pool, out_string = [], []

        for id, file in enumerate(files):
            out_string.append('%-80s' % file)
            feature_pool.append(get_metadata(file))
            out_string[id] += '%6s' % feature_pool[id][0]
            if type == 'grid':
                feature_vector, feature_index, weight = self.read_feature(file, raw = True)
            if type == 'random':
                junk, feature_vector = self.create_synthetic_data(output = file, raw = True)
            for value in feature_vector:
                out_string[id] += '%12f' % value
            out_string[id] += '\n'
        out_string = permutation(out_string)

        ## \todo { Add a check to make sure not to override old data file carelessly }
        with open('../data/' + type + '-' + self.feature_name + '.txt', 'w') as out:
            for i in range(len(out_string)):
                out.write(out_string[i])

        if type == 'grid':
            self.feature_grid = feature_pool
        if type == 'random':
            self.feature_random = feature_pool

    ## Train a machine learning model
    # train a classifier beased on feature file.
    def train(self):
        data = genfromtxt('../data/grid-' + self.feature_name + '.txt')
        data_x = data[:, 2:]
        data_id = genfromtxt('../data/grid-' + self.feature_name + '.txt', usecols = (0,), dtype = str)
        data_y = genfromtxt('../data/grid-' + self.feature_name + '.txt', usecols = (1,), dtype = str)
        self.classifier.fit(data_x, data_y)
        print 'training score = ', self.classifier.score(data_x, data_y)
        self.data_x = data_x
        self.data_y = data_y
        self.data_id = data_id

    ## 5 fold insample validation
    def insample_validation(self):
        score = cross_val_score(self.classifier, self.data_x, self.data_y, cv = 5)
        print 'insample validation score = ', mean(score)
        return score

    ## outsample validation based on random data
    def outsample_validation(self):
        data = genfromtxt('../data/random-' + self.feature_name + '.txt')
        data_x = data[:, 2:]
        data_y = genfromtxt('../data/random-' + self.feature_name + '.txt', usecols = (1,), dtype = str)
        print 'outsample validation score = ', self.classifier.score(data_x, data_y)

    ## classify the dynamic parameterization
    def classify(self, data):
        print self.classifier.predict(data)

    ## guess the value of the statevector on a coarse grid
    def guess(self, y_obs):
        type = self.classifier.predict(y_obs)[0]
        id = arange(len(self.data_y))[self.data_y == type]
        data_x = self.data_x[id]
        data_file = self.data_id[id]
        data_y = array([get_metadata(file)[1:] for file in data_file])
        result = []
        n_bags, n_id = 1000, len(id)
        index = randint(n_id, size = (n_bags, n_id))
        for i in range(n_bags):
            self.guessor.fit(data_x[index[i]], data_y[index[i]])
            result.append(self.guessor.predict(y_obs)[0])
        result = mean(array(result), axis = 0)
        return type, result

    ## Perform retrieval based on observations
    # @param y_obs Observations with error
    def inverse(self, 
            y_obs):
        type, guess = self.guess(y_obs)
        p = Parameters()
        with open(self.report, 'w') as file:
            file.write('** Juno Retrieval Begin **\n')
            file.write('type = %s, initial guess = ' % type)
            for i in range(len(guess)): file.write('%f ' % guess[i])
            file.write('\n')
        print type, guess
        p.add('NH3', value = guess[0], min = 0.1, max = 20.)
        p.add('H2O', value = guess[1], min = 0.1, max = 20.)
        if type == 'sat':
            p.add('stretch', value = 1., vary = False)
            result = minimize(self.loss_function, p, iter_cb = self.write_iter_report,
                    col_deriv = True, Dfun = self.jacobian, args = (y_obs, 'f'),
                    xtol = 1.E-4, ftol = 1.E-8, maxfev = 100)
        elif type == 'stretch':
            p.add('stretch', value = guess[2], min = 0.8, max = 5.)
            result = minimize(self.loss_function, p, iter_cb = self.write_iter_report,
                    col_deriv = True, Dfun = self.jacobian, args = (y_obs, 'f'),
                    xtol = 1.E-4, ftol = 1.E-8, maxfev = 100)
        else:
            raise ValueError('Model type not understood')

        with open(self.report, 'a') as file:
            file.write(fit_report(p))
            file.write('\n')
            # write jacobian
            #file.write('Jacobian = \n')
            #jacob = self.jacobian(p, y_obs)
            #m, n = jacob.shape
            #for i in range(m):
            #    for j in range(n):
            #        file.write('%16.4e' % jacob[i, j])
            #    file.write('\n')
            file.write('** Juno Retrieval End **\n')
        return type, parameter_to_list(p)

    ## Loss function
    # @param params JunoRetrieval uses lmfit as the backend for doing inversions. This argument is the object or Parameters in lmfit.
    # @param y_obs Observations with error.
    # @param output Forward model output filename
    def loss_function(self, 
            params, 
            y_obs, 
            output = 'f'):
        if output == 'f': # use params to run a forward model
            self.model.forward(parameter_to_list(params), 'f')
            y, fi, weight = self.read_feature('f')
        else: # do not run forward model, read model result from output
            y, fi, weight = self.read_feature(output)
        return weight * (y - y_obs)

    ## Calculate Jacobian
    # @param params JunoRetrieval uses lmfit as the backend for doing inversions. This argument is the object or Parameters in lmfit.
    # @param y_obs Observations with error.
    # @param output Forward model output filename
    def jacobian(self, 
            params, 
            y_obs, 
            output = None):
        n_params = 0
        for k, key in enumerate(params.keys()):
            if params[key].vary == True:
                print 'key = ', key
                n_params += 1
                pp = parameter_to_list(params)
                pp[k] = params[key].value * (1 - self.eps)
                name = '_%sm' % key
                self.model.prepare(pp, name)

                pp = parameter_to_list(params)
                pp[k] = params[key].value * (1 + self.eps)
                name = '_%sp' % key
                self.model.prepare(pp, name)
        self.model.forward(threads = 2 * n_params)

        result = []
        for key in params.keys():
            if params[key].vary == True:
                yp = self.loss_function(params, y_obs, output = '_%sp' % key)
                ym = self.loss_function(params, y_obs, output = '_%sm' % key)
                #print yp, ym, yp - ym
                #print params[key].value, self.eps
                result.append((yp - ym) / (2 * self.eps * params[key].value))
        return array(result)

    ## Write inversion iteration report after every step
    # @param params JunoRetrieval uses lmfit as the backend for doing inversions. 
    # This argument is the object or Parameters in lmfit.
    # @param iter Iteration id.
    # @param res Residual array.
    # @param *args Argument list passed to loss_function/jacobian.
    # @param **kwargs Keyword arguments, usually not used.
    def write_iter_report(self, 
            params, 
            iter, 
            res, 
            *args, 
            **kwargs):
        y_obs = args[0]
        with open(self.report, 'a') as file:
            rmsd = sqrt(sum(res**2 / len(res)))
            file.write('Step #%i: Root-mean-square deviation= %10.2E\n' % (iter, rmsd))
            file.write(80 * '-' + '\n')
            file.write('Observation:')
            for i in range(len(y_obs)):
                file.write('%12.4f' % y_obs[i])
            file.write('\nResidual:')
            for i in range(len(y_obs)):
                file.write('%12.4E' % res[i])
            file.write('\n')
            for key, info in params.items(): 
                file.write('%s = %12.4E\n' % (key, info.value))
            file.write(80 * '-' + '\n')
