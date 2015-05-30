#! /usr/bin/env python2.7
from JunoCore import *

## @file jacobian-bt.py Calculate and plot jacobian for brightness temperature observations

# forward model is only used for getting wavelength
fwd_file = '../exp/fwd-3.0-3.0-1.0.h5'
jacobian_files = [
        'jacobian-3.0-3.0-1.0-moist-H2O-bt.txt',
        'jacobian-3.0-3.0-1.0-moist-bt.txt',
        'jacobian-3.0-3.0-1.0-H2O-bt.txt',
        'jacobian-3.0-3.0-1.0-bt.txt']
styles = ['-', '--', '-.', ':']
label  = ['Moist+Opacity', 'Moist', 'Opacity', 'None']

model = JunoAtmosphere()
model.read(fwd_file, raw = True)
wavelength = 30. / model.freq

figure(1, figsize = (12, 10))
ax = axes()
hlegend = []

for i in range(len(jacobian_files)):
    data = genfromtxt(jacobian_files[i])
    h, = ax.plot(wavelength, data[:44, 0], 'g' + styles[i], label = label[i])
    ax.plot(wavelength, data[:44, 1], 'b' + styles[i])
    hlegend.append(h)

ax.set_ylabel('Jacobian', fontsize = 20)
#ax.set_yscale('log')
#ax.set_ylim(-5E-3, 0.015)
ax.set_xscale('log')
ax.set_xlabel('Wavelength (cm)', fontsize = 20)
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,20,30,40,50])
ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10,20,30,40,50])
ax.set_xlim([0, 50])
ax.minorticks_on()
ax.tick_params(width = 2, length = 8)
ax.tick_params(width = 2, which = 'minor', length = 4)
grid(True, which = 'major', color = 'k', linestyle = '--')
grid(True, which = 'minor', color = '0.3', linestyle = ':')
ax.legend(loc = 3, fontsize = 20, ncol = 2)
savefig('../figure/jacobian-bt.png', bbox_inches = 'tight')
#show()


