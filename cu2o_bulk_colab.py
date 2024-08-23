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
  
def CuD_FCC111(bulk, n_layers, vacuum): #not updated
    slab = cu2o111(bulk, n_layers, vacuum)
    unsat_Cu_z = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    mask2=(slab.positions[:, 2] >= unsat_Cu_z) & (slab.symbols=='Cu')
    del slab[mask2]
    #bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    #mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    #slab.set_constraint(FixAtoms(mask=mask1))
    return slab

def STO_FCC111(bulk, n_layers, vacuum):
    slab = cu2o111(bulk, n_layers, vacuum)
    superslab=make_supercell(slab, [[2,-1,0], [-1,2, 0],  [0,0,1]] )
    STO_z = np.max(superslab[superslab.symbols=='O'].positions[:,2])
    mask2 = (superslab.positions[:, 2] >= STO_z) & (superslab.symbols=='O')
    index_to_remove = np.argmax(mask2)
    del atoms[index_to_remove]
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return superslab
 
def CuDO_FCC111(bulk, n_layers, vacuum): #not updated
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
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return superslab

def cu2o100(bulk, n_layers, vacuum):
  #bulk.translate(np.array([1.0, 1.0, 1.0])*1.0)
  #bulk.wrap()
  slab = surface(bulk, (1,0,0), n_layers, vacuum=vacuum, periodic=True) 
  return slab 
  
def Oterm1x1(bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    bottom_Cu_z = np.min(slab[slab.symbols=='Cu'].positions[:,2])
    mask1=slab.positions[:, 2] < bottom_Cu_z + 1.0
    slab.set_constraint(FixAtoms(mask=mask1))
    return slab
    
def Cuterm1x1 (bulk, n_layers, vacuum):
    slab = cu2o100(bulk, n_layers, vacuum)
    #CuMax = np.max(slab[slab.symbols=='Cu'].positions[:,2])
    #mask2=(slab.positions[:, 2] >= CuMax) & (slab.symbols=='Cu')
    #del slab[mask2]
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
    
def hollow_STNO3(bulk,n_layers,vacuum,Cl_X_position,Cl_Y_position,Cl_Z_position): ##untested function
    slab_initial = cu2o111(bulk, n_layers, vacuum)
    slab= make_supercell(slab_initial, [[4,0,0], [0,4, 0],  [0,0,1]])
    N_X_position=0
    N_Y_position=0
    N_Z_position=10
    O_X_position=1.4
    O_Y_position=0
    O_Z_position=10

    Max_O= np.max(slab[slab.symbols=='O'].positions[:,2])
    Max_Cu= np.max(slab[slab.symbols=='Cu'].positions[:,2])
    N_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [N_X_position,N_Y_position, N_Z_position] 
    O_pos = np.mean(slab.positions[slab.positions[:,2] > (Max_Cu and Max_O).all(), :], axis=0) + [O_X_position,O_Y_position, O_Z_position] 
    N = Atoms(symbols='N', positions = [N_pos])
    O_ads = Atoms(symbols='O', positions = [O_pos])

    Ob= O_ads.copy()
    Ob.translate([-1.4,1.3,0])
    Oc= O_ads.copy()
    Oc.translate([-2.4,-1,0])

    slabads = slab + O_ads + Ob + Oc + N
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

 
def bulk_identifier(slab, cutoff_distance=10.0):
    if type(slab) is not Atoms:
        raise Exception('Invalid input. Please provide an Atoms object for your desired slab model')
    # Sort the slab based on z-coordinate
    slab = sort(slab.copy(), np.array(slab.get_positions()[:, 2]).tolist())
    a, b, c = slab.cell  # Extract cell vectors

    # Calculate all pairwise distances between atoms
    distances = slab.get_all_distances()
    z_coord_values = []
    cutoff_distance = cutoff_distance

    for atom_i in slab:
        for atom_j in slab:
            if atom_i.symbol == atom_j.symbol and atom_i.index != atom_j.index:
                diff_xy = np.array(atom_j.position[:2] - atom_i.position[:2])
                cell_vec_xy = np.array(slab.cell[:2, :2])
                int_vec = np.linalg.solve(cell_vec_xy.transpose(), diff_xy)

                if is_close_to_integer(int_vec).all():
                    # further check is done to see if the coordination environment of the atom j is similar ot atom i
                    # based on cutoff distance.
                    cutoff_obeyed_i = [dist for ind, dist in enumerate(distances[atom_i.index].tolist())
                                       if
                                       dist <= cutoff_distance and slab[ind].position[2] / atom_i.position[2] >= 0.9999]
                    cutoff_obeyed_j = [dist for ind, dist in enumerate(distances[atom_j.index].tolist())
                                       if
                                       dist <= cutoff_distance and slab[ind].position[2] / atom_j.position[2] >= 0.9999]

                    if len(cutoff_obeyed_j) == len(cutoff_obeyed_i):
                        dist_diff = np.linalg.norm(np.array(cutoff_obeyed_j)) / np.linalg.norm(
                            np.array(cutoff_obeyed_i))
                        if 0.99 <= dist_diff <= 1.01:
                            z_dist = atom_j.position[2] - atom_i.position[2]
                            z_coord_values.append(z_dist)

    z_coord_values = np.array(z_coord_values)
    slab.center()
    # set the cell parameters to the original 'a' and 'b' whereas the 'c' vector is changed to a new value which
    # represents the minimum repeating unit in the z-direction
    slab.set_cell(np.array([a, b, [0, 0, min(np.abs(z_coord_values))]]))

    # Modify positions to lie within the cell
    x_coord = slab.get_positions()[:, 0]
    y_coord = slab.get_positions()[:, 1]
    new_z_coord = slab.get_positions()[:, 2] - min(slab.get_positions()[:, 2])
    slab.set_positions(np.array([x_coord, y_coord, new_z_coord]).transpose())

    # Remove atoms beyond cell boundaries
    atoms_to_del = [atom_i.index for atom_i in slab if atom_i.position[2] > slab.cell[2, 2]]
    new_bulk = slab[[i for i in range(len(slab)) if i not in atoms_to_del]]

    return new_bulk # return the new bulk model
    
def is_close_to_integer(arr, tolerance=1e-2):
    diff = np.abs(arr - np.round(arr))
    close_to_integer = diff < tolerance
    return close_to_integer

