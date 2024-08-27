from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms, UnitCellFilter
from ase.optimize import LBFGS
from ase.io import read, write
from ase.optimize import MDMin
import numpy as np, sys, os
import pandas as pd
from tabulate import tabulate
from cu2o_bulk_colab import cu2o_bulk, cu2o111, hollow_STCl, atop_unsatCu_STCl, atop_satCu_STCl
from ase.calculators.aims import Aims

from carmm.run.aims_path import set_aims_command
from carmm.run.aims_calculator import get_aims_and_sockets_calculator

jid = os.environ['SLURM_JOB_ID']
print ("sys.argv = ", sys.argv)
set_aims_command(hpc="hawk", basis_set="light", defaults=2020)

aims_dir=f'/scratch/{os.environ["USER"]}/tmp_aims_{jid}'
print (f'{aims_dir=}')
os.mkdir(aims_dir)

n_layers = int(sys.argv[1])


socket_calc, fhi_calc = get_aims_and_sockets_calculator(dimensions=0, logfile=f'{aims_dir}/socketio.log')
fhi_calc.directory = aims_dir

fhi_calc.parameters.pop("xc")
fhi_calc.set(override_warning_libxc="true", # <---- necessary !!!
       #override_warning_libxc="True",
       xc='libxc MGGA_X_MBEEF+GGA_C_PBE_SOL',
       #xc_pre=['pbe', '50'],
       spin='none', # if any(init_magmoms) != 0 else 'none',
       k_grid=(4,4,1),   # to be used in a 3x2 cell
       relativistic=('atomic_zora','scalar'),
       #compensate_multipole_errors='True',
       use_dipole_correction='True',
       #many_body_dispersion= "True",
       compute_forces="true",
       #compute_analytical_stress="true",
       mixer='pulay',
       #charge_mix_param=0.05,
       occupation_type='gaussian 0.01',
       #sc_accuracy_etot=1e-5,
       #sc_accuracy_forces=1e-3,
       #sc_accuracy_rho=5e-3,
       sc_iter_limit=300)


bulk = read('bulk_rlxdft.xyz')
E_bulk= bulk.get_potential_energy()


#hollow 
slabads = hollow_STCl(bulk, n_layers=n_layers, vacuum=10,Cl_X_position=0,Cl_Y_position=0, Cl_Z_position=4)
write('hollowSTCl.xyz', slabads)
print(f'{slabads=}')
slabads.calc = socket_calc
E_slab = slabads.get_potential_energy()
print(f'{n_layers=} {E_slab=}')
'''
qn = LBFGS(slabads, trajectory='Cu2O111.traj')
write('hollowSTCl.traj', slabads)
qn.run(fmax=0.01)
E_slab_qn = slabads.get_potential_energy()
  
#E_surf = (E_slab_qn - E_bulk * n_layers) / 2 / np.linalg.det(slabads.cell[:2, :2])
print(np.linalg.det(slabads.cell[:2, :2]))
print(f'{n_layers=} {E_surf=}')
'''

'''
#atop unsat Cu
slabads = atop_unsatCu_STCl(bulk, n_layers=n_layers, vacuum=10)
write('hollowSTCl.xyz', slabads)
print(f'{slabads=}')
slabads.calc = socket_calc
E_slab = slabads.get_potential_energy()
print(f'{n_layers=} {E_slab=}')

qn = LBFGS(slabads, trajectory='atopUnsat.traj')
write('atopUnsat.traj', slabads)
qn.run(fmax=0.01)
E_slab_qn = slabads.get_potential_energy()
  
#E_surf = (E_slab_qn - E_bulk * n_layers) / 2 / np.linalg.det(slabads.cell[:2, :2])
print(np.linalg.det(slabads.cell[:2, :2]))
print(f'{n_layers=} {E_surf=}')
'''

'''
#atop sat Cu
slabads = atop_satCu_STCl(bulk, n_layers=n_layers, vacuum=10)
write('hollowSTCl.xyz', slabads)
print(f'{slabads=}')
slabads.calc = socket_calc
E_slab = slabads.get_potential_energy()
print(f'{n_layers=} {E_slab=}')

qn = LBFGS(slabads, trajectory='atopSat.traj')
write('atopSat.traj', slabads)
qn.run(fmax=0.01)
E_slab_qn = slabads.get_potential_energy()
  
#E_surf = (E_slab_qn - E_bulk * n_layers) / 2 / np.linalg.det(slabads.cell[:2, :2])
print(np.linalg.det(slabads.cell[:2, :2]))
print(f'{n_layers=} {E_surf=}')
'''
'''
#CuO bridge
slab = cu2o111(bulk, n_layers=n_layers, vacuum=10)
#write('super_ST.xyz', slab)
print(f'{slab=}')
slab.calc = socket_calc
E_slab = slab.get_potential_energy()
print(f'{n_layers=} {E_slab=}')


qn = LBFGS(slab, trajectory='Cu2O111.traj')
write('Cu2O111.traj', slab)
qn.run(fmax=0.01)
E_slab_qn = slab.get_potential_energy()
  
E_surf = (E_slab_qn - E_bulk * n_layers) / 2 / np.linalg.det(slab.cell[:2, :2])
print(f'{n_layers=} {E_surf=}')
'''

