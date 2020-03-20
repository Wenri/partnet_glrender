import sys
from itertools import product, chain
from typing import Final

import numpy as np
import pyximport
from pcl import PointCloud, GeneralizedIterativeClosestPoint
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

from matlabengine import Minboundbox, ICP_finite

pyximport.install(language_level=3)

from pcmetric import pclsimilarity


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class DominantIncomplete(Error):
    """Exception raised for errors in the dominant procedures.

    Attributes:
        components -- components matrix in which the error occurred
    """

    def __init__(self, components):
        self.components = components


def arr2pt(array) -> PointCloud:
    ptcloud = PointCloud()
    ptcloud.from_array(np.asarray(array, dtype=np.float32))
    return ptcloud


def diag_dominant(components_, strict=False):
    maxidx = np.argmax(np.abs(components_), axis=1)
    while True:
        u, idx, counts = np.unique(maxidx, return_index=True, return_counts=True)
        if u.size == maxidx.size:
            break
        if strict or u.size < maxidx.size - 1:
            raise DominantIncomplete(components_)
        else:
            print(f'Dominant Incomplete in relax mode', file=sys.stderr)
            dup, = u[counts > 1]
            missing, = set(range(maxidx.size)) - set(u)
            pending = np.flatnonzero(maxidx == dup)
            value = np.argmax(np.abs(components_[pending, missing]))
            maxidx[pending[value]] = missing

    components_ = components_[idx, :]
    components_ *= np.sign(np.diagonal(components_))

    assert np.abs(np.linalg.det(components_) - 1) < 1e-4

    return components_


class AxisAlign(object):
    def __init__(self, a, pca_approx=True):
        self._components = None
        self._minbbox = None
        self.mean = None
        self._corner_points = None

        if pca_approx:
            pca = PCA()
            pca.fit(np.asarray(a))
            self.mean, self._components = pca.mean_, pca.components_
        else:
            self.mean = np.mean(np.asarray(a), axis=0)

        if not pca_approx:
            self._minbbox = Minboundbox(a)

    def _eval_results(self):
        rot_matrix, self._corner_points = self._minbbox
        self._components = rot_matrix.T

    @property
    def components(self):
        if self._components is None:
            self._eval_results()
        return self._components

    @property
    def corner_points(self):
        if self._corner_points is None:
            self._eval_results()
        return self._corner_points

    def transform(self, a):
        return (a - self.mean) @ self.components.T


def generate_rotmatrix():
    def m_xyz(deg):
        return tuple(R.from_euler(axis, deg, degrees=True).as_matrix() for axis in 'xyz')

    def chaindedup(mat_list):
        mats = set(np.asarray(a, dtype=np.int).tobytes('C') for a in mat_list)
        return tuple(np.reshape(np.frombuffer(buf, dtype=np.int), (3, 3), order='C') for buf in mats)

    base = chaindedup(chain.from_iterable(m_xyz(a) for a in (0, 90, 180, 270)))
    level2 = chaindedup(np.matmul(y, x) for x, y in product(base, repeat=2))
    level3 = chaindedup(np.matmul(y, x) for x, y in product(base, level2))

    return level3


class PCMatch(object):
    _ROT_MATRIX: Final = generate_rotmatrix()

    def __init__(self, *arrays):
        self.arrays = list(arrays)
        self._saved_arrays = []
        self._staged_array = None
        self.is_modified = False

    def center_match(self):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)

        ptloc = np.max(ptarray, axis=0) + np.min(ptarray, axis=0)
        pmloc = np.max(pmarray, axis=0) + np.min(pmarray, axis=0)
        offset = (ptloc - pmloc) / 2
        pmarray += offset

        self.arrays[1] = pmarray
        return offset

    def scale_match(self, coaxis=False):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)

        ax = None
        if coaxis:
            ax = AxisAlign(np.concatenate(self.arrays, axis=0), pca_approx=True)
            print(f'{ax.components=}')
            ptarray = ptarray @ ax.components.T
            pmarray = pmarray @ ax.components.T

        ptlen = np.max(ptarray, axis=0) - np.min(ptarray, axis=0)
        pmlen = np.max(pmarray, axis=0) - np.min(pmarray, axis=0)
        scale = ptlen / pmlen

        pmarray *= scale

        scale = np.diag(scale)
        if ax is not None:
            pmarray = pmarray @ ax.components
            scale = scale @ ax.components

        self.arrays[1] = pmarray
        return scale, self.center_match()

    def axis_match(self, *arrayid):
        aligns = [AxisAlign(self.arrays[i], pca_approx=False) for i in arrayid]

        for id, align in zip(arrayid, aligns):
            self.arrays[id] = align.transform(self.arrays[id])
            print(f'{id=} {align.components=}, {align.mean=}')

        return aligns

    def icp_match(self):
        ptcloud, pmcloud = (arr2pt(np.asarray(a)) for a in self.arrays)

        icp = GeneralizedIterativeClosestPoint()
        converged, transf, estimate, fitness = icp.gicp(pmcloud, ptcloud)
        print(f'{converged=}, {transf=}, {fitness=}')

        self.arrays[1] = np.asarray(estimate)
        return transf, fitness

    def icpf_match(self, registration='Affine'):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)
        estimate, transf = ICP_finite(ptarray, pmarray, Registration=registration)
        print(f'{transf=}')

        self.arrays[1] = np.asarray(estimate)
        return transf

    def rotmatrix_match(self):
        pmarray = self.arrays[1]
        sim_min = np.Inf
        trans = None
        offset = None
        pm_min = None
        for matrix in self._ROT_MATRIX:
            self.arrays[1] = pmarray @ matrix.T
            scale, off = self.scale_match(coaxis=False)
            sim = self.similarity()
            if sim < sim_min:
                sim_min = sim
                pm_min = self.arrays[1]
                trans, offset = scale @ matrix, off
            print(f'similarity {sim=}')
        self.arrays[1] = pm_min
        return sim_min, trans, offset

    def register(self, n_iters=10):
        for i in range(n_iters):
            with self as step:
                with step as txn1, step as txn2:
                    txn1.scale_match(coaxis=True)
                    with txn1 as txn1a:
                        txn1a.icp_match()
                    print('Optional scale&icp_match is done.')
                    txn2.icp_match()
                print('Optional scale_match/icp_match is done.')
                with step as txn1, step as txn2:
                    txn1.icpf_match(registration='Resize')
                    txn2.icpf_match(registration='Affine')
                print('Optional icpf_match is done.')
            print(f'iter {i} is done.')
            if not self.is_modified:
                break
        return self.similarity()

    def similarity(self):
        clouds = (arr2pt(a) for a in self.arrays)
        return pclsimilarity(*clouds)

    def __enter__(self):
        new_pcm = PCMatch(*self.arrays)
        if not self._saved_arrays:
            self.is_modified = False
        self._saved_arrays.append(new_pcm)
        return new_pcm

    def __exit__(self, exc_type, exc_val, exc_tb):
        saved_pcm = self._saved_arrays.pop()

        if self._staged_array is None or saved_pcm.similarity() < self._staged_array.similarity():
            self._staged_array = saved_pcm

        if not self._saved_arrays:
            if self._staged_array.similarity() < self.similarity():
                self.arrays = self._staged_array.arrays
                self.is_modified = True
            else:
                print('Transaction is not better, discarding...')
            self._staged_array = None

