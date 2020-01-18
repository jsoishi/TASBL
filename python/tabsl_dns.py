import time
from configparser import ConfigParser
from pathlib import Path
import sys
import logging

import numpy as np

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools


from filter_field_iso import filter_field

logger = logging.getLogger(__name__)

# Parses .cfg filename passed to script
config_file = Path(sys.argv[-1])

# Parse .cfg file to set global parameters for script
runconfig = ConfigParser()
runconfig.read(str(config_file))

datadir = Path("runs") / config_file.stem
# Parameters
params = runconfig['params']
nx = params.getint('nx')
ny = params.getint('ny')
nz = params.getint('nz')
Lx = params.getfloat('Lx')
Ly = params.getfloat('Ly')

Re = params.getfloat('Re')
Pr = params.getfloat('Pr')
Ra = params.getfloat('Ra')
tau = params.getfloat('tau') # characteristic scale for mask
ampl = params.getfloat('ampl') # IC amplitude
use_Laguerre = params.getboolean('Laguerre')

run_params = runconfig['run']
stop_wall_time = run_params.getfloat('stop_wall_time')
stop_sim_time = run_params.getfloat('stop_sim_time')
stop_iteration = run_params.getint('stop_iteration')
dt = run_params.getfloat('dt')

threeD = True
if nx == 0:
    threeD = False
    logger.info("Running in 2D")
else:
    logger.info("Running in 3D")
# Create bases and domain
start_init_time = time.time()
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
if use_Laguerre:
    logger.info("Running with Laguerre z-basis")
    z_basis = de.Laguerre('z', nz, dealias=3/2)
else:
    Lz = params.getfloat('Lz')
    logger.info("Running with Chebyshev z-basis, Lz = {}".format(Lz))
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)
bases = [y_basis, z_basis]
if threeD:
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    bases.insert(0,x_basis)

domain = de.Domain(bases, grid_dtype=np.float64)
if domain.dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir()

problem = de.IVP(domain, variables=['p','θ','u','v','w','θz','uz','vz','wz'], time='t')
problem.parameters['Ra'] = Ra
problem.parameters['Re'] = Re
problem.parameters['Pr'] = Pr
problem.substitutions['Uz'] = '-Pr'
if use_Laguerre:
    problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))'
    problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
    #problem.substitutions['Ux'] = '-Pr*Re*exp(-z)'
    #problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
    problem.substitutions['T0'] = 'exp(-Pr*z)'
    problem.substitutions['T0z'] = '-Pr*exp(-Pr*z)'
else:
    problem.parameters['Lz'] = Lz
    problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))/(1-exp(-Lz))'
    problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)/(1-exp(-Lz))'
    problem.substitutions['T0'] = '(exp(-Pr*z) - exp(-Pr*Lz))/(1-exp(-Pr*Lz))'
    problem.substitutions['T0z'] = '-Pr*exp(-Pr*z)/(1-exp(-Pr*Lz))'
problem.substitutions['udotgradU_x'] = 'w*Uxz'

if threeD:
    problem.substitutions['Lap(A,Az)'] = 'dx(dx(A)) + dy(dy(A)) + dz(Az)'
    problem.substitutions['Udotgrad(A)'] = 'Ux*dx(A) + Uz*dz(A)'
    problem.substitutions['udotgrad(A,Az)'] = 'u*dx(A) + v*dy(A) + w*Az'
else:
    problem.substitutions['Lap(A,Az)'] = 'dy(dy(A)) + dz(Az)'
    problem.substitutions['Udotgrad(A,Az)'] = 'Uz*Az'
    problem.substitutions['udotgrad(A,Az)'] = 'v*dy(A) + w*Az'

if threeD:
    problem.add_equation("dx(u) + dy(v) + wz = 0")
    problem.add_equation("dt(u) - Pr*Lap(u,uz) + dx(p)     = -udotgrad(u,uz) - udotgradU_x - Udotgrad(u,uz) ")
else:
    problem.add_equation("dy(v) + wz = 0")
    problem.add_equation("dt(u) - Pr*Lap(u,uz) + udotgradU_x + Udotgrad(u,uz) = -udotgrad(u,uz) ")

problem.add_equation("dt(v) - Pr*Lap(v,vz) + dy(p)  + Udotgrad(v,vz) = -udotgrad(v,vz) ")
problem.add_equation("dt(w) - Pr*Lap(w,wz) + dz(p)  + Udotgrad(w,wz) - Pr*Ra*θ  = -udotgrad(w,wz) ")
problem.add_equation("dt(θ) - Lap(θ,θz) + Udotgrad(θ,θz) + w*T0z = - udotgrad(θ,θz) ")

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
problem.add_bc("left(w) = 0", condition="(ny != 0)")
problem.add_bc("left(p) = 0", condition="(ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
logger.info("L (ky=0) condition number: {:e}".format(np.linalg.cond((solver.pencils[0].L+solver.pencils[0].M).A)))
logger.info("L (ky=1) condition number: {:e}".format(np.linalg.cond((solver.pencils[1].L+solver.pencils[1].M).A)))

# Initial conditions
if threeD:
    z = domain.grid(2)
else:
    z = domain.grid(1)
θ = solver.state['θ']
θz = solver.state['θz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

### Initial conditions
# Thermal: backtground + perturbations damped at wall
# noise is dense in Laguerres even when damped.
# instead, for now just do a few sine modes.
y = domain.grid(0)
modes = [10,2,3,5]
noise = np.zeros_like(θ['g'])

for m in modes:
    noise += np.sin(2*np.pi*m*y/Ly)

n = 40
#mask = ((1-np.cos(2*np.pi/Lz * (z+30)))/2)**n
# this mask uses only 2 laguerres!
#mask = z/tau * np.exp(1-z/tau)
mask = z/tau * (np.exp(1-z/tau) - np.exp(1-Lz/tau))
θ['g'] = ampl * noise * mask
θ.differentiate('z', out=θz)
# w = solver.state['w']
# v = solver.state['v']
# wz = solver.state['wz']
# vz = solver.state['vz']
# θ.differentiate('z', out=w)
# θ.differentiate('y', out=v)
# v['g'] *= -1
# v.differentiate('z',out=vz)
# w.differentiate('z',out=wz)
# θ['g'] = 0.

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Analysis
# snap = solver.evaluator.add_file_handler('snapshots', sim_dt=0.2, max_writes=10)
# if threeD:
#     snap.add_task("interp(p, x=0)", scales=1, name='p midplane')
#     snap.add_task("interp(θ, x=0)", scales=1, name='θ midplane')
#     snap.add_task("interp(u, x=0)", scales=1, name='u midplane')
#     snap.add_task("interp(v, x=0)", scales=1, name='v midplane')
#     snap.add_task("interp(w, x=0)", scales=1, name='w midplane')
# else:
#     snap.add_task("p", scales=1, name='p')
#     snap.add_task("θ", scales=1, name='θ')
#     snap.add_task("u", scales=1, name='u')
#     snap.add_task("v", scales=1, name='v')
#     snap.add_task("w", scales=1, name='w')
analyses = []
check = solver.evaluator.add_file_handler(datadir / Path('checkpoints'),iter=500,max_writes=100)
check.add_system(solver.state)
analyses.append(check)
check_c = solver.evaluator.add_file_handler(datadir / Path('checkpoints_c'),iter=500,max_writes=100)
check_c.add_system(solver.state, layout='c')
analyses.append(check_c)
timeseries = solver.evaluator.add_file_handler(datadir / Path('timeseries'), iter=100)
timeseries.add_task("integ(0.5*(u*u + v*v + w*w))", name='KE')
analyses.append(timeseries)
# CFL

## CFL doesn't work for Laguerres yet because they don't have a grid difference method...
#CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=1.5,
#                     max_change=1.5, min_change=0.5, max_dt=0.05)
#CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("0.5*sqrt(u*u + v*v + w*w)", name='Ke')
flow.add_property("w", name='w')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        #dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Kin En = %e' %flow.max('Ke'))
            logger.info('Max w = %e' %flow.max('w'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('beginning join operation')
    for task in analyses:
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)

    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

