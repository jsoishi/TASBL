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
ky = 0.
nz = 500
Lz = 50

Re = 0.
Pr = 1
Ra = 20000

z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)

domain = de.Domain([z_basis], comm=MPI.COMM_SELF)

problem = de.EVP(domain, variables=['θ','Ψ','phi','θz','Ψz','phiz','phizz','phizzz'], eigenvalue='sigma')
problem.parameters['Ra'] = Ra
problem.parameters['Re'] = Re
problem.parameters['Pr'] = Pr
problem.parameters['kx'] = kx
problem.parameters['ky'] = ky
problem.substitutions['k2'] = '(kx*kx + ky*ky)'
problem.substitutions['Uz'] = '-Pr'
problem.parameters['Lz'] = Lz
problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))/(1-exp(-Lz))'
problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)/(1-exp(-Lz))'
problem.substitutions['Uxzz'] = '-Pr*Re*exp(-z)/(1-exp(-Lz))'
problem.substitutions['T0'] = 'Ra*(exp(-Pr*z) - exp(-Pr*Lz))/(1-exp(-Pr*Lz))'
problem.substitutions['T0z'] = '-Pr*Ra*exp(-Pr*z)/(1-exp(-Pr*Lz))'

problem.add_equation('sigma*Ψ/Pr + 1j*kx*Ux*Ψ/Pr - Ψz - dz(Ψz) + k2*Ψ + 1j*ky*Uxz*phi/Pr = 0')
problem.add_equation('sigma*phizz/Pr + 1j*kx*Ux*phizz/Pr - phizzz - k2*sigma*phi/Pr - 1j*kx*k2*Ux*phi/Pr + k2*phiz - 1j*kx*Uxzz*phi/Pr - dz(phizzz) + 2*k2*phizz - k2*k2*phi + θ = 0')
problem.add_equation('sigma*θ + 1j*kx*Ux*θ - Pr*θz - dz(θz) + k2*θ + k2*T0z*phi = 0.')

problem.add_equation("θz - dz(θ) = 0")
problem.add_equation("Ψz - dz(Ψ) = 0")
problem.add_equation("phiz - dz(phi) = 0")
problem.add_equation("phizz - dz(phiz) = 0")
problem.add_equation("phizzz - dz(phizz) = 0")

problem.add_bc("right(θ) = 0")
problem.add_bc("right(Ψ) = 0")
problem.add_bc("right(phi) = 0")
problem.add_bc("right(phiz) = 0")

problem.add_bc("left(θ) = 0")
problem.add_bc("left(Ψ) = 0")
problem.add_bc("left(phi) = 0")
problem.add_bc("left(phiz) = 0")

EP = Eigenproblem(problem,sparse=True)

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
mins = np.array((10, 0.25))
maxs = np.array((50, 0.75))
nums = np.array((10  , 10))
try:
    cf.load_grid('TASBL_Re0_growth_rates.h5')
except:
    cf.grid_generator(mins, maxs, nums)
    if comm.rank == 0:
        cf.save_grid('TASBL_Re0_growth_rates')
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


