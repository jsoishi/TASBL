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
z_basis = de.Laguerre('z', nz, dealias=3/2)
bases = [y_basis, z_basis]
if threeD:
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    bases.insert(0,x_basis)

domain = de.Domain(bases, grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','θ','u','v','w','θz','uz','vz','wz'], time='t')
problem.parameters['Ra'] = Ra
problem.parameters['Re'] = Re
problem.parameters['Pr'] = Pr
problem.substitutions['Ux'] = 'Pr*Re*(1-exp(-z))'
problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
#problem.substitutions['Ux'] = '-Pr*Re*exp(-z)'
#problem.substitutions['Uxz'] = 'Pr*Re*exp(-z)'
problem.substitutions['Uz'] = '-Pr'
problem.substitutions['T0'] = 'Ra*exp(-Pr*z)'
problem.substitutions['T0z'] = '-Pr*Ra*exp(-Pr*z)'
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
    problem.add_equation("dt(u) - Pr*Lap(u,uz) = -udotgrad(u,uz) - udotgradU_x - Udotgrad(u,uz) ")

problem.add_equation("dt(v) - Pr*Lap(v,vz) + dy(p)           = -udotgrad(v,vz) - Udotgrad(v,vz) ")
problem.add_equation("dt(w) - Pr*Lap(w,wz) + dz(p) - Pr*Ra*θ = -udotgrad(w,wz) - Udotgrad(w,wz) ")
problem.add_equation("dt(θ) - Lap(θ,θz) = -w*T0z - udotgrad(θ,θz)")

problem.add_equation("θz - dz(θ) = 0", tau=False)
problem.add_equation("uz - dz(u) = 0", tau=False)
problem.add_equation("vz - dz(v) = 0", tau=False)
problem.add_equation("wz - dz(w) = 0", tau=False)

problem.add_bc("left(θ) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0", condition="(ny != 0)")
problem.add_bc("left(p) = 0", condition="(ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
logger.info("L (ky=0) condition number: {:e}".format(np.linalg.cond((dt*solver.pencils[0].L+solver.pencils[0].M).A)));
logger.info("L (ky=1) condition number: {:e}".format(np.linalg.cond((dt*solver.pencils[1].L+solver.pencils[1].M).A)));
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

# this mask uses only 2 laguerres!
mask = z/tau * np.exp(1-z/tau)
θ['g'] = ampl * noise * mask
θ.differentiate('z', out=θz)

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
check = solver.evaluator.add_file_handler('checkpoints',iter=100,max_writes=10)
check.add_system(solver.state)
check_c = solver.evaluator.add_file_handler('checkpoints_c',iter=100,max_writes=10)
check_c.add_system(solver.state, layout='c')

# CFL

## CFL doesn't work for Laguerres yet because they don't have a grid difference method...
#CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=1.5,
#                     max_change=1.5, min_change=0.5, max_dt=0.05)
#CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("0.5*sqrt(u*u + v*v + w*w)", name='Ke')



# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        #dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Kin En = %e' %flow.max('Ke'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

