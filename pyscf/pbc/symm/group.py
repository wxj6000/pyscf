#!/usr/bin/env python
# Copyright 2020-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

from abc import ABC, abstractmethod
import numpy as np
from pyscf.pbc.symm import geom
from pyscf.pbc.symm import symmetry

class GroupElement(ABC):
    '''
    The abstract class for group elements.
    '''
    def __call__(self, other):
        return self.__matmul__(other)

    @abstractmethod
    def __matmul__(self, other):
        pass

    def __mul__(self, other):
        assert isinstance(other, self.__class__)
        return self @ other

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def inv(self):
        '''
        Inverse of the group element.
        '''
        pass

class PGElement(GroupElement):
    '''
    The class for crystallographic point group elements.
    The group elements are rotation matrices represented
    in lattice translation vector basis.

    Attributes:
        matrix : (d,d) array of ints
            Rotation matrix in lattice translation vector basis.
        dimension : int
            Dimension of the space: `d`.
    '''
    def __init__(self, matrix):
        self.matrix = matrix
        self.dimension = matrix.shape[0]

    def __matmul__(self, other):
        if not isinstance(other, PGElement):
            raise TypeError(f"{other} is not a point group element.")
        return PGElement(np.dot(self.matrix, other.matrix))

    def __repr__(self):
        return self.matrix.__repr__()

    def __hash__(self):
        def _id(op):
            s = op.flatten() + 1
            return int(''.join([str(i) for i in s]), op.shape[0])

        r = _id(self.rot)
        # move identity to the first place
        d = self.dimension
        r -= _id(np.eye(d, dtype=int))
        if r < 0:
            r += _id(np.ones((d,d), dtype=int)) + 1
        return r

    def __lt__(self, other):
        if not isinstance(other, PGElement):
            raise TypeError(f"{other} is not a point group element.")
        return self.__hash__() < other.__hash__()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, a):
        self._matrix = a

    rot = matrix

    def inv(self):
        return PGElement(np.asarray(np.linalg.inv(self.matrix), dtype=np.int32))

class FiniteGroup():
    '''
    The class for finite groups.

    Attributes:
        elements : list
            Group elements.
        order : int
            Group order.
    '''
    def __init__(self, elements):
        self.elements = np.asarray(elements)
        self._order = None
        self._hash_table = None
        self._inverse_table = None
        self._multiplication_table = None
        self._conjugacy_table = None
        self._conjugacy_mask = None

    def __len__(self):
        return self.order

    def __getitem__(self, i):
        return self.elements[i]

    @property
    def order(self):
        if self._order is None:
            self._order = len(self.elements)
        return self._order

    @order.setter
    def order(self, n):
        self._order = n

    @property
    def hash_table(self):
        '''
        Hash table for group elements: {hash : index}.
        '''
        if self._hash_table is None:
            self._hash_table = {hash(g) : i for i, g in enumerate(self.elements)}
        return self._hash_table

    @hash_table.setter
    def hash_table(self, table):
        self._hash_table = table

    @property
    def inverse_table(self):
        '''
        Table for inverse of the group elements.

        Return : (n,) array of ints
            The indices of elements.
        '''
        if self._inverse_table is None:
            _table = [self.hash_table[hash(g.inv())] for g in self.elements]
            self._inverse_table = np.asarray(_table)
        return self._inverse_table

    @inverse_table.setter
    def inverse_table(self, table):
        self._inverse_table = table

    @property
    def multiplication_table(self):
        '''
        Multiplication table of the group.

        Return : (n, n) array of ints
             The indices of elements.
        '''
        if self._multiplication_table is None:
            prod = self.elements[:,None] * self.elements[None,:]
            _table = [self.hash_table[hash(gh)] for gh in prod.flatten()]
            self._multiplication_table = np.asarray(_table).reshape(prod.shape)
        return self._multiplication_table

    @multiplication_table.setter
    def multiplication_table(self, table):
        self._multiplication_table = table

    @property
    def conjugacy_table(self):
        '''
        conjugacy_table[`index_g`, `index_x`] returns the index of element `h`,
        where :math:`h = x * g * x^{-1}`.
        '''
        if self._conjugacy_table is None:
            prod_table = self.multiplication_table
            g_xinv = prod_table[:,self.inverse_table]
            self._conjugacy_table = prod_table[np.arange(self.order)[None,:], g_xinv]
        return self._conjugacy_table

    @conjugacy_table.setter
    def conjugacy_table(self, table):
        self._conjugacy_table = table

    @property
    def conjugacy_mask(self):
        '''
        Boolean mask array indicating whether two elements
        are conjugate with each other.
        '''
        if self._conjugacy_mask is None:
            n = self.order
            is_conjugate = np.zeros((n,n), dtype=bool)
            is_conjugate[np.arange(n)[:,None], self.conjugacy_table] = True
            self._conjugacy_mask = is_conjugate
        return self._conjugacy_mask

    def conjugacy_classes(self):
        '''
        Compute conjugacy classes.

        Returns:
            classes : (n_irrep,n) boolean array
                The indices of `True` correspond to the
                indices of elements in this class.
            representatives : (n_irrep,) array of ints
                Representive elements' indices in each class.
            inverse : (n,) array of ints
                The indices to reconstruct `conjugacy_mask` from `classes`.
        '''
        _, idx = np.unique(self.conjugacy_mask, axis=0, return_index=True)
        representatives = np.sort(idx)
        classes = self.conjugacy_mask[representatives]
        inverse = -np.ones((self.order), dtype=int)
        diff = (self.conjugacy_mask[None,:,:]==classes[:,None,:]).all(axis=-1)
        for i, a in enumerate(diff):
            inverse[np.where(a==True)[0]] = i # noqa: E712 
        assert (inverse >= 0).all()
        assert (classes[inverse] == self.conjugacy_mask).all()
        return classes, representatives, inverse

    def character_table(self, return_full_table=False):
        '''
        Character table of the group.

        Args:
            return_full_table : bool
                If True, also return character table for all elements.

        Returns:
            chi : array
                Character table for classes.
            chi_full : array, optional
                Character table for all elements.
        '''
        classes, _, inverse = self.conjugacy_classes()
        class_sizes = classes.sum(axis=1)

        ginv_h = self.multiplication_table[self.inverse_table]
        M  = classes @ np.random.rand(self.order)[ginv_h] @ classes.T
        M /= class_sizes

        _, Rchi = np.linalg.eig(M)
        chi = Rchi.T / class_sizes

        norm = np.sum(np.abs(chi) ** 2 * class_sizes[None,:], axis=1) ** 0.5
        chi  = chi / norm[:,None] * self.order ** 0.5
        chi /= (chi[:, 0] / np.abs(chi[:, 0]))[:,None]
        chi  = np.round(chi, 9)
        chi_copy = chi.copy()
        chi_copy[:,1:] *= -1
        idx = np.lexsort(np.rot90(chi_copy))
        chi = chi[idx]
        if return_full_table:
            chi_full = chi[:, inverse]
            return chi, chi_full
        else:
            return chi

class PointGroup(FiniteGroup):
    '''
    The class for crystallographic point groups.
    '''
    def group_name(self, notation='international'):
        name = geom.get_crystal_class(None, self.elements)[0]
        if notation.lower().startswith('scho'): # Schoenflies
            from pyscf.pbc.symm.tables import SchoenfliesNotation
            name = SchoenfliesNotation[name]
        return name


def symm_adapted_basis(cell):
    sym = symmetry.Symmetry(cell).build(symmorphic=True)
    Dmats = sym.Dmats

    elements = []
    for op in sym.ops:
        assert(op.trans_is_zero)
        elements.append(op.rot)

    elements = np.unique(np.asarray(elements), axis=0)
    elements = [PGElement(rot) for rot in elements]
    elements.sort()

    pg = PointGroup(elements)
    chartab = pg.character_table(return_full_table=True)[1]
    nirrep = len(chartab)
    nao = cell.nao
    coords = cell.get_scaled_positions()
    atm_maps = []
    for op in sym.ops:
        atm_map, _ = symmetry._get_phase(cell, op, coords, None, ignore_phase=True)
        atm_maps.append(atm_map)
    atm_maps = np.asarray(atm_maps)
    tmp = np.unique(atm_maps, axis=0)
    tmp = np.sort(tmp, axis=0)
    tmp = np.unique(tmp, axis=1)
    eql_atom_ids = []
    for i in range(tmp.shape[-1]):
        eql_atom_ids.append(np.unique(tmp[:,i]))

    aoslice = cell.aoslice_by_atom()
    cbase = np.zeros((nirrep, nao, nao))
    for atom_ids in eql_atom_ids:
        iatm = atom_ids[0]
        op_relate_idx = []
        for iop in range(pg.order):
            op_relate_idx.append(atm_maps[iop][iatm])
        ao_loc = np.array([aoslice[i,2] for i in op_relate_idx])

        b0, b1 = aoslice[iatm,:2]
        ioff = 0
        icol = aoslice[iatm, 2]
        for ib in range(b0, b1):
            nctr = cell.bas_nctr(ib)
            l = cell.bas_angular(ib)
            if cell.cart:
                degen = (l+1) * (l+2) // 2
            else:
                degen = l * 2 + 1
            for n in range(degen):
                for iop in range(pg.order):
                    Dmat = Dmats[iop][l]
                    tmp = np.einsum('x,y->xy', chartab[:,iop], Dmat[:,n])
                    idx = ao_loc[iop] + ioff
                    for ictr in range(nctr):
                        cbase[:, idx:idx+degen, icol+n+ictr*degen] += tmp / pg.order
                        idx += degen
            ioff += degen * nctr
            icol += degen * nctr

    so = []
    for ir in range(nirrep):
        idx = np.where(np.sum(abs(cbase[ir]), axis=0) > 1e-9)[0]
        so.append(cbase[ir][:,idx])

    for ir in range(nirrep):
        norm = np.linalg.norm(so[ir], axis=0)
        so[ir] /= norm[None,:]
    return so

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = [['O' , (1. , 0.    , 0.   ,)],
                 ['H' , (0. , -.757 , 0.587,)],
                 ['H' , (0. , 0.757 , 0.587,)]]
    cell.a = [[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]
    cell.basis = 'ccpvdz'
    cell.verbose = 5
    cell.build()
    so = symm_adapted_basis(cell)

    from pyscf import gto as mol_gto
    from pyscf.symm import geom as mol_geom
    from pyscf.symm.basis import symm_adapted_basis as mol_symm_adapted_basis
    mol = cell.copy()
    gpname, origin, axes = mol_geom.detect_symm(mol._atom)
    atoms = mol_gto.format_atom(cell._atom, origin, axes)
    mol.build(False, False, atom=atoms)
    mol_so = mol_symm_adapted_basis(mol, gpname)[0]

    print(abs(so[0] - mol_so[0]).max())
    print(abs(so[1] - mol_so[1]).max())
    print(abs(so[2] - mol_so[3]).max())
    print(abs(so[3] - mol_so[2]).max())
