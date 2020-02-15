import os

import numpy as np
import pcl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pcl import GeneralizedIterativeClosestPoint, PointCloud
from sklearn.decomposition import PCA

from cfgreader import conf


def arr_to_ptcloud(array) -> PointCloud:
    ptcloud = PointCloud()
    ptcloud.from_array(np.asarray(array, dtype=np.float32))
    return ptcloud


def cvt_obj2pcd(file, outdir, argv0='pcl_mesh_sampling', **kwargs):
    name, ext = os.path.splitext(os.path.basename(file))
    outfile = os.path.join(outdir, name + '.pcd')
    argv = [argv0, '-no_vis_result']
    for option, value in kwargs.items():
        argv.append(f'-{option}')
        if value:
            argv.append(str(value))
    argv += [file, outfile]
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(f'Spawn {argv0}:', ' '.join(argv))
    return os.spawnvp(os.P_WAIT, argv0, argv)


def cvt_load_pcd(fname, faceid=3, n_samples=10000):
    p_dir = os.path.join(conf.pcd_dir, fname)
    rname = f'parts_render_0.face{faceid}'
    cvt_obj2pcd(os.path.join(conf.data_dir, fname + '.obj'),
                p_dir, n_samples=n_samples, leaf_size=0.001)
    cvt_obj2pcd(os.path.join(conf.pix2mesh_dir, fname, rname + '.obj'),
                p_dir, n_samples=n_samples, leaf_size=0.0005)
    ptcloud = pcl.load(os.path.join(p_dir, fname + '.pcd'))
    pmcloud = pcl.load(os.path.join(p_dir, rname + '.pcd'))

    print(f'{ptcloud.size=}, {pmcloud.size=}')
    return ptcloud, pmcloud


def pca_match(*arrays):
    def arrayPCA(a):
        pca = PCA()
        pca.fit(np.asarray(a))
        return pca

    def diag_dominant(pca: PCA) -> PCA:
        maxidx = np.argmax(np.abs(pca.components_), axis=1)
        u, idx = np.unique(maxidx, return_index=True)
        assert u.size == 3
        pca.components_ = pca.components_[idx, :]
        pca.components_ *= np.sign(np.diagonal(pca.components_))
        return pca

    ptpca, pmpca = (diag_dominant(arrayPCA(a)) for a in arrays)

    ptdet = np.linalg.det(ptpca.components_)
    pmdet = np.linalg.det(pmpca.components_)

    print(f'{ptpca.components_=}')
    print(f'{pmpca.components_=}')
    print(f'{ptdet=} {pmdet=}')

    ptarray, pmarray = arrays
    ptarray -= ptpca.mean_
    ptarray = ptarray @ ptpca.components_.T

    pmarray -= pmpca.mean_
    pmarray = pmarray @ pmpca.components_.T

    ptlen = np.max(ptarray, axis=0) - np.min(ptarray, axis=0)
    pmlen = np.max(pmarray, axis=0) - np.min(pmarray, axis=0)
    scale = ptlen / pmlen

    pmarray *= scale

    return ptarray, pmarray


def icp_match(*arrays):
    ptcloud, pmcloud = (arr_to_ptcloud(a) for a in arrays)

    icp = GeneralizedIterativeClosestPoint()
    converged, transf, estimate, fitness = icp.gicp(pmcloud, ptcloud)
    transfdet = np.linalg.det(transf[:3, :3])
    print(f'{converged=}, {transf=}, {transfdet=}, {fitness=}')

    return np.asarray(ptcloud), np.asarray(pmcloud), transf


def main(idx):
    ptarray, pmarray = pca_match(*cvt_load_pcd(conf.dblist[idx]))
    ptcloud, pmcloud, transf = icp_match(ptarray, pmarray)

    fig = pyplot.figure()
    ax = Axes3D(fig)

    ax.scatter(*ptcloud.T, s=1, marker='.', color='b')
    # ax.scatter(*pmcloud.T, s=1, marker='.', color='r')

    pmcloud = pmcloud @ transf[:3, :3].T
    pmcloud += transf[:3, 3]

    ax.scatter(*pmcloud.T, s=1, marker='.', color='g')
    pyplot.show()


if __name__ == '__main__':
    main(0)
