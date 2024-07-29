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
    
def cu2o_bulk():
    bulk=read('9007497.cif')
    #bulk.translate([5.0,5.0,5.0])
    #bulk.wrap()
    return bulk

def cu2o111(bulk, n_layers, vacuum):
  bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
  bulk.wrap()
  slab = surface(bulk, (1,1,1), n_layers, vacuum=vacuum, periodic=True) 
  bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
  mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
  slab.set_constraint(FixAtoms(mask=mask1))
  return slab 
  
def CuD_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu')
    del slab[mask2]
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return slab

def STO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    return superslab
 
def CuDO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    del superslab[index_to_remove]
    unsat_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2])
    mask3=(superslab.positions[:, 2] >= unsat_Cu_z) & (superslab.symbols=='Cu') 
    del superslab[mask3]
    return superslab
    
def py111(bulk,n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    O_pos = np.mean(superslab.positions[superslab.positions[:,2] > 15, :], axis=0) + [2.35,1.3, 3.75]
    Cu_pos = np.mean(superslab.positions[superslab.positions[:,2] > 15, :], axis=0) + [1.0,0.5, 2.0]
    O = Atoms(symbols='O', positions = [O_pos])
    Cu = Atoms(symbols='Cu', positions = [Cu_pos])
    Cu2 = Cu.copy()
    Cu2.translate([2.8,0,0])
    Cu3 = Cu.copy()
    Cu3.translate([1.3,2.5,0])
    superslab = superslab + O + Cu + Cu2 + Cu3
    return superslab

    
def cu2o100(bulk, n_layers, vacuum):
  #bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
  #bulk.wrap()
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
    Cu_max = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask3=(slab.positions[:, 2] >= Cu_max) & (slab.symbols=='Cu') 
    indices=list()
    for i in range(len(slab)):
        if mask3[i]==True:
            indices.append(i)
    midpoint = np.mean(slab.positions[indices[:]],axis=0)
    v=(midpoint-slab.positions[indices[0]])/2
    slab.positions[indices[0]]+=-v
    slab.positions[indices[1]]+=v
    return slab
    
def c2x2(bulk, n_layers, vacuum):
    slab = Oterm1x1(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[1,1,0], [-1,1, 0],  [0,0,1]] )
    Max_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 2.0
    mask2=(superslab.positions[:, 2] >= Max_Cu_z) & (superslab.symbols=='Cu')
    del superslab[mask2]
    Max_O_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 0.5
    mask3=(superpositions[:, 2] >= Max_O_z) & (superslab.symbols=='O')
    del superslab[mask3]
    return superslab
    
def slab3011(bulk,n_layers,vacuum):
    slab = Oterm1x1(bulk, 4, 10)
    superslab=make_supercell(slab, [[2,-1,0], [1,1, 0], [0,0,1]] )
    Max_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 2.0
    mask2=(superslab.positions[:, 2] >= Max_Cu_z) & (superslab.symbols=='Cu')
    del superslab[mask2]
    return superslab
