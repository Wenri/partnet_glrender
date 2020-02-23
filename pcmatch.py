import sys
from itertools import product, chain

import numpy as np
import pyximport
from pcl import PointCloud, GeneralizedIterativeClosestPoint
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

pyximport.install(language_level=3)

from matlabengine import Minboundbox, ICP_finite


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


def arr_to_ptcloud(array) -> PointCloud:
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
        pca = PCA()
        pca.fit(np.asarray(a))
        self._components = None
        self._mean = pca.mean_
        self._result = None

        if pca_approx:
            try:
                self._components = diag_dominant(pca.components_, strict=True)
            except DominantIncomplete as e:
                print(f'Dominant Incomplete in {e.components}, retrying using MinBBOX', file=sys.stderr)
                pca_approx = False

        if not pca_approx:
            minbbox = Minboundbox()
            self._result = minbbox(a)

    @property
    def components(self):
        if self._components is None:
            self._components = diag_dominant(self._result.result())
        return self._components

    @property
    def mean(self):
        return self._mean


def generate_rotmatrix():
    def m_xyz(deg):
        return (R.from_euler(axis, deg, degrees=True).as_matrix() for axis in 'xyz')

    def m_iter_deg(*deg_list):
        for deg in deg_list:
            yield from m_xyz(deg)
            yield from m_xyz(-deg)

    def chaindedup(*mat_list):
        mats = set(np.asarray(a, dtype=np.int).tobytes('C') for a in chain(*mat_list))
        return [np.reshape(np.frombuffer(buf, dtype=np.int), (3, 3), order='C') for buf in mats]

    base = chaindedup(m_iter_deg(0, 90, 180))
    level2 = chaindedup(np.matmul(y, x) for x, y in product(base, repeat=2))
    level3 = chaindedup(np.matmul(y, x) for x, y in product(level2, base))

    return level3


class PCMatch(object):
    rotmatrix = generate_rotmatrix()

    def __init__(self, *arrays):
        self.arrays = list(arrays)

    def scale_match(self, coaxis=False):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)

        if coaxis:
            ax = AxisAlign(np.concatenate(self.arrays, axis=0), pca_approx=True)
            print(f'{ax.components=}')
            ptarray = ptarray @ ax.components.T
            pmarray = pmarray @ ax.components.T

        ptlen = np.max(ptarray, axis=0) - np.min(ptarray, axis=0)
        pmlen = np.max(pmarray, axis=0) - np.min(pmarray, axis=0)
        scale = ptlen / pmlen

        pmarray *= scale
        self.arrays = [ptarray, pmarray]
        return scale

    def icp_match(self):
        ptcloud, pmcloud = (arr_to_ptcloud(np.asarray(a)) for a in self.arrays)

        icp = GeneralizedIterativeClosestPoint()
        converged, transf, estimate, fitness = icp.gicp(pmcloud, ptcloud)
        print(f'{converged=}, {transf=}, {fitness=}')

        self.arrays[1] = np.asarray(estimate)
        return transf, fitness

    def axis_match(self, *arrayid):
        aligns = [AxisAlign(self.arrays[i], pca_approx=False) for i in arrayid]

        for id, align in zip(arrayid, aligns):
            ptcomp, ptmean = align.components, align.mean
            a = self.arrays[id]

            print(f'{ptcomp=} {id=}')

            a -= ptmean
            self.arrays[id] = a @ ptcomp.T

    def icpf_match(self, registration='Affine'):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)
        icpf = ICP_finite()
        estimate, transf = icpf(ptarray, pmarray, Registration=registration).result()
        print(f'{transf=}')

        self.arrays[1] = np.asarray(estimate)
        return transf
