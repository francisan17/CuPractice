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

aims_dir=f'/scratch/{os.environ["USER"]}/tmp_aims_{jid}'
os.mkdir(aims_dir)

print (f'{aims_dir=}')

sockets_calc, fhi_calc = get_aims_and_sockets_calculator(dimensions=0, logfile=f'{aims_dir}/socketio.log')
fhi_calc.directory = aims_dir

fhi_calc.parameters.pop("xc")
fhi_calc.set(override_warning_libxc="true", # <---- necessary !!!
       xc='libxc MGGA_X_MBEEF+GGA_C_PBE_SOL',
       xc_pre=['pbe', '50'],
       spin='none', # if any(init_magmoms) != 0 else 'none',
       k_grid=(3,3,3),   # to be used in a 3x2 cell
       relativistic=('atomic_zora','scalar'),
       #compensate_multipole_errors='True',
       #use_dipole_correction='True',
       compute_forces="true",
       compute_analytical_stress="true",
       mixer='pulay',
       occupation_type='gaussian 0.01',
       #sc_accuracy_etot=1e-5,
       #sc_accuracy_forces=1e-3,
       #sc_accuracy_rho=5e-3,
       sc_iter_limit=300)


bulk = cu2o_bulk()
bulk.set_calculator(sockets_calc)
ucf = UnitCellFilter(bulk)
LBFGS(ucf).run(fmax=0.01)
E_bulk = bulk.get_potential_energy()
write('Cu2O_rlxdft.xyz', bulk)
