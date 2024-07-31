from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms, UnitCellFilter
from ase.optimize import LBFGS
from ase.io import read, write
from ase.optimize import MDMin
import numpy as np, sys, os
import pandas as pd
from tabulate import tabulate
from cu2o_bulk_colab import cu2o_bulk, cu2o111, STO_FCC111
from ase.calculators.aims import Aims

from carmm.run.aims_path import set_aims_command
from carmm.run.aims_calculator import get_aims_and_sockets_calculator

jid = os.environ['SLURM_JOB_ID']
print ("sys.argv = ", sys.argv)
set_aims_command(hpc="hawk", basis_set="light", defaults=2020)

#aims_dir=f'/scratch/{os.environ["USER"]}/tmp_aims_{jid}'
aims_dir=f'/scratch/{os.environ["USER"]}/tmp_aims'

print (f'{aims_dir=}')


sockets_calc, fhi_calc = get_aims_and_sockets_calculator(dimensions=0, logfile=f'{aims_dir}/socketio.log')
fhi_calc.directory = aims_dir

fhi_calc.parameters.pop("xc")
fhi_calc.set(override_warning_libxc="true", # <---- necessary !!!
       #override_warning_libxc="True",
       xc='libxc MGGA_X_MBEEF+GGA_C_PBE_SOL',
       xc_pre=['pbe', '50'],
       spin='none', # if any(init_magmoms) != 0 else 'none',
       k_grid=(4,4,4),   # to be used in a 3x2 cell
       relativistic=('atomic_zora','scalar'),
       #compensate_multipole_errors='True',
       #use_dipole_correction='True',
       compute_forces="true",
       compute_analytical_stress="true",
       mixer='pulay',
       #charge_mix_param=0.05,
       occupation_type='gaussian 0.01',
       sc_accuracy_etot=1e-5,
       sc_accuracy_forces=1e-3,
       sc_accuracy_rho=5e-3,
       sc_iter_limit=300)


bulk = cu2o_bulk()
bulk.set_calculator(sockets_calc)
ucf = UnitCellFilter(bulk)
LBFGS(ucf).run(fmax=0.01)
E_bulk = bulk.get_potential_energy()

sys.exit(0)



bulk_oxygens = bulk[bulk.symbols=='O']
#write('bulk.xyz', bulk)

n_layers = { 3,4,5,6,7,8,9,10}
Cu2Otable={}

n_layers_list=[]
slab_oxygens_list=[]
slab_coppers_list=[]
E_surf_list=[]


for n_layers in range(3,11):
  vacuum=10
  superslab = STO_FCC111(bulk, n_layers, vacuum)
  write(f'STO_{n_layers}.traj',superslab)
  slab_oxygens = superslab[superslab.symbols=='O']
  slab_coppers = superslab[superslab.symbols=='Cu']
 #print(f"Number of Oxygen atoms in slab: {len(slab_oxygens)}")

  superslab.calc = mace_mp(model="large", dispersion=True, default_dtype="float64", device='cuda')
  E_slab = superslab.get_potential_energy()
  E_cleav = (E_slab - E_bulk * n_layers) / 2 / np.linalg.det(superslab.cell[:2, :2])
  print(f'{n_layers=} {E_cleav=}')
'''
  qn = MDMin(superslab, trajectory=f'STO_{n_layers}.traj')
  qn.run(fmax=0.01)
  E_slab = superslab.get_potential_energy()

  ##Calc Surface Energy
  E_surf = (E_slab - E_bulk * n_layers) / 2 / np.linalg.det(superslab.cell[:2, :2])
  #print(f'{n_layers=} {E_surf=}')

  n_layers_list.append(n_layers)
  slab_oxygens_list.append(len(slab_oxygens))
  slab_coppers_list.append(len(slab_coppers))
  E_surf_list.append(E_surf)

df = pd.DataFrame({'Number of Layers': n_layers_list,'Number of Copper Atoms':slab_coppers_list, 'Number of Oxygen Atoms':slab_oxygens_list, 'Surface Energy': E_surf_list})
table=tabulate(df, headers = 'keys', tablefmt = 'fancy_grid')
df.to_csv('STO.csv', index=False)
print(table)
'''
