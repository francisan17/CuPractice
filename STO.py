from ase import Atoms
from mace.calculators import mace_mp
from ase.calculators.emt import EMT
from ase.build import surface
from ase.constraints import FixAtoms, UnitCellFilter
from ase.optimize import QuasiNewton
from ase.visualize import view
from ase.io import read, write
from ase.optimize import MDMin
import numpy as np
import pandas as pd
from tabulate import tabulate
from cu2o_bulk_colab import cu2o_bulk, cu2o111, STO_FCC111

bulk = cu2o_bulk()
bulk.calc = mace_mp(model="large", dispersion=True, default_dtype="float64", device='cuda')
ucf = UnitCellFilter(bulk)
MDMin(ucf).run(fmax=0.01)
E_bulk = bulk.get_potential_energy()
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
