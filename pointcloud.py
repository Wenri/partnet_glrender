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


def diag_dominant(components_):
    maxidx = np.argmax(np.abs(components_), axis=1)
    u, idx = np.unique(maxidx, return_index=True)
    if u.size < maxidx.size:
        raise DominantIncomplete(components_)
    components_ = components_[idx, :]
    components_ *= np.sign(np.diagonal(components_))
    return components_


def axis_align(a):
    pca = PCA()
    pca.fit(np.asarray(a))

    a -= pca.mean_
    try:
        components_ = diag_dominant(pca.components_)
    except DominantIncomplete as e:
        print(f'Dominant Incomplete in {e.components}, retring using MinBBOX', file=sys.stderr)
        minbbox = Minboundbox()
        components_ = diag_dominant(minbbox(a).result())

    return components_, pca.mean_


class PCMatch(object):
    def __init__(self, *arrays):
        self.arrays = list(arrays)

    def scale_match(self):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)

        ptlen = np.max(ptarray, axis=0) - np.min(ptarray, axis=0)
        pmlen = np.max(pmarray, axis=0) - np.min(pmarray, axis=0)
        scale = ptlen / pmlen

        pmarray *= scale
        self.arrays[1] = pmarray
        return scale

    def icp_match(self):
        ptcloud, pmcloud = (arr_to_ptcloud(np.asarray(a)) for a in self.arrays)

        icp = GeneralizedIterativeClosestPoint()
        converged, transf, estimate, fitness = icp.gicp(pmcloud, ptcloud)
        print(f'{converged=}, {transf=}, {fitness=}')

        self.arrays[1] = np.asarray(estimate)
        return transf, fitness

    def axis_match(self, arrayid):
        a = self.arrays[arrayid]
        ptcomp, ptmean = axis_align(a)

        ptdet = np.linalg.det(ptcomp)

        print(f'{ptcomp=}')
        print(f'{ptdet=}')

        assert np.abs(ptdet - 1) < 1e-4

        a -= ptmean
        self.arrays[arrayid] = a @ ptcomp.T

        return ptcomp, ptmean

    def icpf_match(self):
        ptarray, pmarray = (np.asarray(a) for a in self.arrays)
        icpf = ICP_finite()
        estimate, transf = icpf(ptarray, pmarray, Registration='Affine').result()
        print(f'{transf=}')

        self.arrays[1] = np.asarray(estimate)
        return transf

def main(idx):
    pcm = PCMatch(*cvt_load_pcd(conf.dblist[idx]))

    MatlabEngine.start()

    pcm.axis_match(0)

    for epoch in range(10):
        pcm.axis_match(1)

        for i in range(10):
            scale = pcm.scale_match()
            transf, fitness = pcm.icp_match()
            if np.sum(np.abs(transf - np.eye(4))) < 1e-4:
                break

        for i in range(10):
            scale = pcm.scale_match()
            transf = pcm.icpf_match()
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
