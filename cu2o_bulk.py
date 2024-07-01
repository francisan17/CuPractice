from ase.io import read, write
from ase import Atoms
from mace.calculators import mace_mp
from ase.build import surface
from ase.visualize import view
from ase.constraints import FixAtoms, UnitCellFilter
from ase.build import bulk, fcc100
from ase.build.supercells import make_supercell
import numpy as np
from ase.optimize import LBFGS

'''
if False:
    x = read('/home/lana/Downloads/9007497.cif')
    x.calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')
    ucf = UnitCellFilter(x)
    qn = LBFGS(ucf, trajectory='bulk_rlx.traj')
    qn.run(fmax=0.01)
    E_x = x.get_potential_energy()
    t = read('bulk_rlx.traj@:')
    view(t)
    '''
    
def cu2o_bulk():
    bulk=read('/home/lana/Downloads/9007497.cif')
    #bulk.translate([5.0,5.0,5.0])
    #bulk.wrap()
    return bulk

def cu2o111(bulk, n_layers, vacuum):
  bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
  bulk.wrap()
  slab = surface(bulk, (1,1,1), n_layers, vacuum=vacuum, periodic=True) 
  return slab 
  
def CuD_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu')
    del slab[mask2]
    return slab

def STO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    slab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(slab[slab.symbols=='O'].positions[:,2])
    mask2 = (slab.positions[:, 2] >= STO_z) & (slab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    print (index_to_remove)
    slab[index_to_remove].symbol='Au'
    del slab[index_to_remove]
    return slab
    
def CuDO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    slab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(slab[slab.symbols=='O'].positions[:,2])
    mask2 = (slab.positions[:, 2] >= STO_z) & (slab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    del slab[index_to_remove]
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask3=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu') 
    del slab[mask3]
    return slab
    
def py111(bulk,n_layers, vacuum):
    slab = CuD_FCC111(bulk, n_layers, vacuum)
    slab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    O_pos = np.mean(slab.positions[slab.positions[:,2] > 15, :], axis=0) + [2.35,1.3, 3.75]
    Cu_pos = np.mean(slab.positions[slab.positions[:,2] > 15, :], axis=0) + [1.0,0.5, 2.0]
    O = Atoms(symbols='O', positions = [O_pos])
    Cu = Atoms(symbols='Cu', positions = [Cu_pos])
    Cu2 = Cu.copy()
    Cu2.translate([2.8,0,0])
    Cu3 = Cu.copy()
    Cu3.translate([1.3,2.5,0])
    slab = slab + O + Cu + Cu2 + Cu3
    return slab
    
def cu2o100(bulk, n_layers, vacuum):
  bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
  bulk.wrap()
  slab = surface(bulk, (1,0,0), n_layers, vacuum=vacuum, periodic=True) 
  return slab 
  
def Cuterm1x1(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    return slab
    
def Oterm1x1(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    CuMax = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= CuMax) & (slab.symbols=='Cu')
    del slab[mask2]
    return slab
    
def dimer1x1(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    CuMax = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= CuMax) & (slab.symbols=='Cu')
    del slab[mask2]
    return slab
    
def ridgedimerc2x2(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    return slab
