from ase.io import read, write
from ase import Atoms
from mace.calculators import mace_mp
from ase.build import surface
from ase.visualize import view
from ase.constraints import FixAtoms, UnitCellFilter
from ase.build import bulk, fcc100
from ase.build.supercells import make_supercell
import numpy as np
from ase.build.tools import sort
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
  
def CuD_FCC111(bulk, n_layers, vacuum): #oxygens too high, an easy fix though!
    slab = cu2o111(bulk, n_layers, vacuum)
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu')
    del slab[mask2]
    return slab

def STO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    del superslab[index_to_remove]
    
    return superslab
 
def CuDO_FCC111(bulk, n_layers, vacuum): #not checked thoroughly but seems close if not quite perfect
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    del superslab[index_to_remove]
    unsat_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2])
    mask3=(superslab.positions[:, 2] >= unsat_Cu_z) & (superslab.symbols=='Cu') 
    del superslab[mask3]
    bottom_Cu_z = np.min(superslab[superslab.symbols=='Cu'].positions[:,2])
    mask1=superslab.positions[:, 2] < bottom_Cu_z + 1.0
    superslab.set_constraint(FixAtoms(mask=mask1))
    return superslab
    
def py111(bulk,n_layers, vacuum):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab=make_supercell(slab_initial, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    O_pos = np.mean(slab.positions[slab.positions[:,2] > 15, :], axis=0) + [2.35,1.3, 3.75]
    Cu_pos = np.mean(slab.positions[slab.positions[:,2] > 15, :], axis=0) + [1.0,0.5, 2.0]
    O = Atoms(symbols='O', positions = [O_pos])
    Cu = Atoms(symbols='Cu', positions = [Cu_pos])
    Cu2 = Cu.copy()
    Cu2.translate([2.8,0,0])
    Cu3 = Cu.copy()
    Cu3.translate([1.3,2.5,0])
    superslab = slab + O + Cu + Cu2 + Cu3
    bottom_Cu_z = np.min(superslab[superslab.symbols=='Cu'].positions[:,2])
    mask1=superslab.positions[:, 2] < bottom_Cu_z + 1.0
    superslab.set_constraint(FixAtoms(mask=mask1))
    return superslab

def cu2o100(bulk, n_layers, vacuum):
  slab = surface(bulk, (1,0,0), n_layers, vacuum=vacuum, periodic=True) 
  bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
  mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
  slab.set_constraint(FixAtoms(mask=mask1))
  return slab 
  
def Oterm1x1(bulk, n_layers, vacuum): ##
    slab = cu2o100(bulk, n_layers, vacuum)
    CuMax = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= CuMax) & (slab.symbols=='Cu')
    del slab[mask2]
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return slab
    
def Cuterm1x1 (bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    OMax = np.max(slab[slab.symbols=='O'].positions[:,2])
    mask2=(slab.positions[:, 2] >= OMax) & (slab.symbols=='O')
    del slab[mask2]
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return slab
    
def hollow_Cuterm (bulk,n_layers,vacuum,Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial= Cuterm1x1 (bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def slab3011(bulk,n_layers,vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [1,1, 0], [0,0,1]] )
    #Max_Cu_z = np.max(superslab[superslab.symbols=='Cu'].positions[:,2]) - 2.0
    #mask2=(superslab.positions[:, 2] >= Max_Cu_z) & (superslab.symbols=='Cu')
    #del superslab[mask2]
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return superslab

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
    mask3=(superslab.positions[:, 2] >= Max_O_z) & (superslab.symbols=='O')
    del superslab[mask3]
    return superslab
    
def Cl_ads(Cl_X_position, Cl_Y_position):
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, 5]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    return slab_ads

def Bridge_STCl_CuO(bulk,n_layers,vacuum,Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def ST4x4(bulk,n_layers,vacuum):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    return slab
    
def Cuterm4x4 (bulk,n_layers,vacuum):
    slab_initial= Cuterm1x1 (bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    return slab

def Bridge_STCl_CuCu(bulk,n_layers,vacuum,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    Cl_X_position=-1.25
    Cl_Y_position=0.9
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def STNO3(bulk,n_layers,vacuum,N_Z_position,O_Z_position): ##untested function, keep Z positions the same.
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    
    N_X_position=0
    N_Y_position=0
    O_X_position=1.4
    O_Y_position=0
    
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    N_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [N_X_position,N_Y_position,N_Z_position] 
    O_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [O_X_position,O_Y_position,O_Z_position] 
    N = Atoms(symbols='N', positions = [N_pos])
    O_ads = Atoms(symbols='O', positions = [O_pos])

    Ob= O_ads.copy()
    Ob.translate([-1.4,1.3,0])
    Oc= O_ads.copy()
    Oc.translate([-2.4,-1,0])

    NO3= O_ads+Ob+Oc+N
    slabads = slab + NO3
    bottom_Cu_z = np.min(slabads[slabads.symbols=='Cu'].positions[:,2])
    mask1=slabads.positions[:, 2] < bottom_Cu_z + 1.0
    slabads.set_constraint(FixAtoms(mask=mask1))
    return slabads

def CutermNO3(bulk,n_layers,vacuum,N_Z_position,O_Z_position): ##untested function, keep Z positions the same.
    slab_initial = Cuterm1x1(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    
    N_X_position=0
    N_Y_position=0
    O_X_position=1.4
    O_Y_position=0
    
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    N_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [N_X_position,N_Y_position,N_Z_position] 
    O_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [O_X_position,O_Y_position,O_Z_position] 
    N = Atoms(symbols='N', positions = [N_pos])
    O_ads = Atoms(symbols='O', positions = [O_pos])

    Ob= O_ads.copy()
    Ob.translate([-1.4,1.3,0])
    Oc= O_ads.copy()
    Oc.translate([-2.2,-0.75,0])

    NO3= O_ads+Ob+Oc+N
    slabads = slab + NO3
    bottom_Cu_z = np.min(slabads[slabads.symbols=='Cu'].positions[:,2])
    mask1=slabads.positions[:, 2] < bottom_Cu_z + 1.0
    slabads.set_constraint(FixAtoms(mask=mask1))
    return slabads
    
def atop_satCu_STCl(bulk,n_layers,vacuum,Cl_Z_position):
    Cl_X_position=2.3
    Cl_Y_position=-0.5
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def atop_unsatCu_STCl(bulk,n_layers,vacuum,Cl_Z_position):
    Cl_X_position=0.75
    Cl_Y_position=-3
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads

def CuObridge1_STCl(bulk,n_layers,vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [-1.343,-0.2885, 4]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads
    
def CuObridge2_STCl(bulk,n_layers,vacuum,Cl_X_position,Cl_Y_position,Cl_Z_position):
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    Cl_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [Cl_X_position,Cl_Y_position, Cl_Z_position]
    Cl = Atoms(symbols='Cl', positions = [Cl_pos])
    Cl_ads = Atoms(symbols='Cl', positions = [Cl_pos])
    slabads = slab + Cl_ads
    return slabads

