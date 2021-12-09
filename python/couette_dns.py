import time
from configparser import ConfigParser
from pathlib import Path
import sys
import logging
import numpy as np

import dedalus.public as de
from dedalus.tools import post
from dedalus.extras import flow_tools

logger = logging.getLogger(__name__)
debug = False

# Parsing .cfg passed
config_file = Path(sys.argv[-1])
logger.info("Running with config file {}".format(str(config_file)))

# Setting global params
runconfig = ConfigParser()
runconfig.read(str(config_file))
datadir = Path("couette_runs") / config_file.stem

# Params
params = runconfig['params']
nx = params.getint('nx')
ny = params.getint('ny')
nz = params.getint('nz')
Lx = params.getfloat('Lx')
Lz = params.getfloat('Lz')

ampl = params.getfloat('ampl')
Re = params.getfloat('Re')

run_params = runconfig['run']
restart = run_params.get('restart_file')
stop_wall_time = run_params.getfloat('stop_wall_time')
stop_sim_time = run_params.getfloat('stop_sim_time')
stop_iteration = run_params.getint('stop_iteration')
dt = run_params.getfloat('dt')

# Create bases and domain?
start_init_time = time.time()
dealias = 3/2
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias = dealias)
z_basis = de.Chebyshev('z', nz, interval = (0,Lz), dealias = dealias)

bases = [x_basis, z_basis]

domain = de.Domain(bases, grid_dtype = np.float64)
if domain.dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir()

problem = de.IVP(domain, variables = ['p', 'u', 'w', 'uz', 'wz'], time = 't')
problem.parameters['Re'] = Re
problem.substitutions['Lap(A, Az)'] = 'dx(dx(A)) + dz(Az)'
problem.substitutions['KE'] = '0.5 * (u**2 + w**2)'
problem.substitutions['U'] = 'z'
problem.substitutions['KE_pert'] = '0.5 * ((u-U)**2 + w**2)'
problem.add_equation('dz(u) - uz = 0')
problem.add_equation('dz(w) - wz = 0')
problem.add_equation('dt(u) + dx(p) - (1/Re)*Lap(u, uz) = -u*dx(u) - w*dz(u)')
problem.add_equation('dt(w) + dz(p) - (1/Re)*Lap(w, wz) = -u*dx(w) - w*dz(w)')
problem.add_equation('dx(u) + wz = 0')
problem.add_bc("right(u) = 1")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0", condition = "nx != 0")
problem.add_bc("right(p) = 0", condition = "nx == 0")


solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info("Solver built")

z = domain.grid(1)

solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = stop_iteration

# Initial conditions
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState()
noise = rand.standard_normal(gshape)[slices]

psi = domain.new_field()
psi['g'] = ampl*noise*np.sin(np.pi*z)
psi.set_scales(0.25, keep_data = True)
psi['c']
psi['g']
psi.set_scales(1, keep_data = True)

u = solver.state['u']
u.set_scales(3/2)
w = solver.state['w']
w.set_scales(3/2)

z = domain.grid(1, scales=dealias)
u['g'] = -psi.differentiate('z')['g'] + z
w['g'] = psi.differentiate('x')['g']

analyses = []
check = solver.evaluator.add_file_handler(datadir / Path('checkpoints'), iter=500, max_writes=100)
check.add_system(solver.state)
analyses.append(check)
check_c = solver.evaluator.add_file_handler(datadir / Path('checkpoints_c'),iter=500,max_writes=100)
check_c.add_system(solver.state, layout='c')
analyses.append(check_c)
timeseries = solver.evaluator.add_file_handler(datadir / Path('timeseries'), iter=100)
timeseries.add_task('integ(KE)', name='KE')
timeseries.add_task('integ(KE_pert)', name = 'KE_pert')
analyses.append(timeseries)

flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("w", name='w')
flow.add_property('KE', name='KE')
flow.add_property('KE_pert', name = 'KE_pert')

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
            logger.info('Max w = %e' %flow.max('w'))
            logger.info('Max KE = %e' %flow.max('KE'))
            logger.info('Max KE_pert = %e' %flow.max('KE_pert'))
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
    
