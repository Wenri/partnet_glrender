import os
import sys

import numpy as np
import pcl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pcl import GeneralizedIterativeClosestPoint, PointCloud
from sklearn.decomposition import PCA

from cfgreader import conf
from matlabengine import Minboundbox, ICP_finite, MatlabEngine


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


def cvt_obj2pcd(file, outdir, argv0='pcl_mesh_sampling', **kwargs):
    if not os.path.exists(file):
        return -1
    name, ext = os.path.splitext(os.path.basename(file))
    outfile = os.path.join(outdir, name + '.pcd')
    argv = [argv0, '-no_vis_result', '-write_normals']
    for option, value in kwargs.items():
        argv.append(f'-{option}')
        if value:
            argv.append(str(value))
    argv += [file, outfile]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(f'Spawn {argv0}:', ' '.join(argv))
    return os.spawnvp(os.P_WAIT, argv0, argv)


def cvt_load_pcd(fname, faceid=3, after_merging=True, n_samples=5000):
    p_dir = os.path.join(conf.pcd_dir, fname)
    ret = cvt_obj2pcd(os.path.join(conf.data_dir, fname + '.obj'),
                      p_dir, n_samples=n_samples, leaf_size=0.001)
    assert ret == 0

    rname = f'parts_render_after_merging_0.face{faceid}'
    ret = cvt_obj2pcd(os.path.join(conf.pix2mesh_dir, fname, rname + '.obj'),
                      p_dir, n_samples=n_samples, leaf_size=0.0005) if after_merging else -1

    if ret != 0:
        rname = f'parts_render_0.face{faceid}'
        ret = cvt_obj2pcd(os.path.join(conf.pix2mesh_dir, fname, rname + '.obj'),
                          p_dir, n_samples=n_samples, leaf_size=0.0005)

    ptcloud = pcl.load(os.path.join(p_dir, fname + '.pcd'))
    pmcloud = pcl.load(os.path.join(p_dir, rname + '.pcd'))

    print(f'{ptcloud.size=}, {pmcloud.size=}')
    ptcloud, pmcloud = np.asarray(ptcloud), np.asarray(pmcloud)
    return ptcloud, pmcloud


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


class PCMatch(object):
    def __init__(self, *arrays):
        self.arrays = list(arrays)

    def scale_match(self):
        ax = AxisAlign(np.concatenate(self.arrays, axis=0))
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)
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


def main(idx):
    pcm = PCMatch(*cvt_load_pcd(conf.dblist[idx]))

    MatlabEngine.start()

    pcm.axis_match(0, 1)

    for epoch in range(5):
        for i in range(10):
            scale = pcm.scale_match()
            transf, fitness = pcm.icp_match()
            if np.sum(np.abs(transf - np.eye(4))) < 1e-4:
                break

        for i in range(10):
            scale = pcm.scale_match()
            transf = pcm.icpf_match(registration='Affine')
            if np.sum(np.abs(transf - np.eye(4))) < 1e-4:
                break

    ptarray, pmarray = (np.asarray(a) for a in pcm.arrays)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(*ptarray.T, s=1, marker='.', color='g')
    ax.scatter(*pmarray.T, s=1, marker='.', color='r')

    pyplot.show()


if __name__ == '__main__':
    main(20)
