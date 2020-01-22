import sys
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

df = Path(sys.argv[-1])
outdir = df.parents[1]

data = h5py.File(df,'r')

t = data['scales/sim_time'][:]
KE= data['tasks/KE'][:,0,0]
try:
    Nu = data['tasks/Nu'][:,0,0]
except KeyError:
    Nu = False


if Nu is not False:
    plt.subplot(211)
    plt.plot(t,KE)
    plt.xlabel('time')
    plt.ylabel('Kinetic Energy')
    plt.subplot(212)
    plt.plot(t,Nu)
    plt.xlabel('time')
    plt.ylabel('Nu')
    plt.savefig(outdir/Path("KE_Nu_vs_t.png"), dpi=100)
else:
    plt.plot(t,KE)
    plt.xlabel('time')
    plt.ylabel('Kinetic Energy')
    plt.savefig(outdir/Path("KE_vs_t.png"), dpi=100)
