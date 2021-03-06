from mpi4py import MPI
import time
from pathlib import Path
import sys
import logging
from eigentools import Eigenproblem, CriticalFinder
import numpy as np

import dedalus.public as de

comm = MPI.COMM_WORLD

logger = logging.getLogger(__name__)

kx = 0.
ky = 0.438
nz = 250
Lz = 50

Re = 0.
Pr = 1
Ra = 20
use_Laguerre = False
find_crit = True

if use_Laguerre:
    logger.info("Running with Laguerre z-basis")
    z_basis = de.Laguerre('z', nz, dealias=3/2)
else:
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)

domain = de.Domain([z_basis])

problem = de.EVP(domain, variables=['p','θ','u','v','w','θz','uz','vz','wz'], eigenvalue='sigma')
problem.parameters['Ra'] = Ra
problem.parameters['Re'] = Re
problem.parameters['Pr'] = Pr
problem.parameters['kx'] = kx
problem.parameters['ky'] = ky
problem.substitutions['dt(A)'] = '2*sigma*A'
problem.substitutions['dx(A)'] = '1j*kx*A'
problem.substitutions['dy(A)'] = '1j*ky*A'
if use_Laguerre:
    problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))'
    problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
    #problem.substitutions['Ux'] = '-Pr*Re*exp(-z)'
    #problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
    problem.substitutions['T0'] = 'exp(-Pr*z)'
    problem.substitutions['T0z'] = '-Ra*exp(-Pr*z)'
else:
    problem.parameters['Lz'] = Lz
    problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))/(1-exp(-Lz))'
    problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)/(1-exp(-Lz))'
    problem.substitutions['T0'] = 'Ra*(exp(-Pr*z) - exp(-Pr*Lz))/(1-exp(-Pr*Lz))'
    problem.substitutions['T0z'] = '-Pr*Ra*exp(-Pr*z)/(1-exp(-Pr*Lz))'

problem.substitutions['Lap(A,Az)'] = 'dx(dx(A)) + dy(dy(A)) + dz(Az)'

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(u)/(Pr*Ra) - Uxz*w/(Pr*Ra) - dx(p) + 2*Lap(u,uz)/Ra = 0")
problem.add_equation("dt(v)/(Pr*Ra)                 - dy(p) + 2*Lap(v,vz)/Ra = 0.")
problem.add_equation("dt(w)/(Pr*Ra) - Uxz*u/(Pr*Ra) - dz(p) + 2*Lap(w,wz)/Ra - θ*T0z + θ = 0.")

problem.add_equation("dt(θ) - w*T0z + w + 2*Lap(θ,θz) = 0.")

if use_Laguerre:
    problem.add_equation("θz - dz(θ) = 0", tau=False)
    problem.add_equation("uz - dz(u) = 0", tau=False)
    problem.add_equation("vz - dz(v) = 0", tau=False)
    problem.add_equation("wz - dz(w) = 0", tau=False)
else:
    problem.add_equation("θz - dz(θ) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("vz - dz(v) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("right(θ) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(v) = 0")
    problem.add_bc("right(w) = 0")
    
problem.add_bc("left(θ) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")

EP = Eigenproblem(problem,sparse=True)

if find_crit:
    def shim(x,y):
        gr, indx, freq = EP.growth_rate({"Ra":x,"ky":y})
        ret = gr+1j*freq
        if type(ret) == np.ndarray:
            return ret[0]
        else:
            return ret

    cf = CriticalFinder(shim, comm)

    # generating the grid is the longest part
    start = time.time()
    mins = np.array((1, 0.25))
    maxs = np.array((20, 0.75))
    nums = np.array((10  , 10))
    try:
        cf.load_grid('TASBL_enstab_primative_Re0.h5')
    except:
        cf.grid_generator(mins, maxs, nums)
        if comm.rank == 0:
            cf.save_grid('TASBL_enstab_primative_Re0')
    end = time.time()
    if comm.rank == 0:
        print("grid generation time: {:10.5f} sec".format(end-start))

    cf.root_finder()
    crit = cf.crit_finder(find_freq = True)

    if comm.rank == 0:
        print("crit = {}".format(crit))
        print("critical wavenumber alpha = {:10.5f}".format(crit[1]))
        print("critical Re = {:10.5f}".format(crit[0]))
        print("critical omega = {:10.5f}".format(crit[2]))

        cf.save_grid('orr_sommerfeld_growth_rates')
        cf.plot_crit()
else:
    gr, idx, freq = EP.growth_rate({})
    print("TASBL growth rate = {0:10.5e}".format(gr))

