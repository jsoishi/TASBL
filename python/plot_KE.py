import sys
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

df = Path(sys.argv[-1])
outdir = df.parents[2]

data = h5py.File(df,'r')

t = data['scales/sim_time'][:]
KE= data['tasks/KE'][:,0,0]
plt.plot(t,KE)
plt.xlabel('time')
plt.ylabel('Kinetic Energy')
plt.savefig(outdir/Path("KE_vs_t.png"), dpi=100)
