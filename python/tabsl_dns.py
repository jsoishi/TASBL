import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = (25., 25., 1.)
nx, ny ,nz = (64, 64, 16)
epsilon = 0.8
Pr = 1.0
Re = 10.
Ra = 17

# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Laguerre('z', nz, dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], time='t')
problem.parameters['Ra'] = Ra
problem.parameters['Re'] = Re
problem.parameters['Pr'] = Pr

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(b) - dx(dx(b)) + dy(dy(b)) + dz(bz)             = - u*dx(b) - v*dy(b) - w*bz")
problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)     = - u*dx(u) - v*dy(u) - w*uz")
problem.add_equation("dt(v) - Pr*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)     = - u*dx(v) - v*dy(v) - w*vz")
problem.add_equation("dt(w) - Pr*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - Pr*Ra*b = - u*dx(w) - v*dy(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(b) = Ra")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = -Pr")
problem.add_bc("right(b) = 0")
problem.add_bc("right(u) = Pr*Re")
problem.add_bc("right(v) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("right(w) = -Pr")
problem.add_bc("left(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

# Initial conditions
z = domain.grid(2)
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['v']
wz = solver.state['vz']
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

### Initial conditions

# Velocity
u['g'] = Pr*Re*(1-np.exp(-z))
u.differentiate('z', out=uz)

w['g'] = -Pr

# Thermal: backtground + perturbations damped at wall
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
b['g'] = Ra*np.exp(-Pr*z)
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = 100
solver.stop_wall_time = 60 * 60.
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=0.2, max_writes=10)
snap.add_task("interp(p, z=0)", scales=1, name='p midplane')
snap.add_task("interp(b, z=0)", scales=1, name='b midplane')
snap.add_task("interp(u, z=0)", scales=1, name='u midplane')
snap.add_task("interp(v, z=0)", scales=1, name='v midplane')
snap.add_task("interp(w, z=0)", scales=1, name='w midplane')
snap.add_task("integ(b, 'z')", name='b integral x4', scales=4)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=1.5,
                     max_change=1.5, min_change=0.5, max_dt=0.05)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

