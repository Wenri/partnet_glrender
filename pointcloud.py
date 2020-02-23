import os

import numpy as np
import pcl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from cfgreader import conf
from matlabengine import MatlabEngine
from pcmatch import PCMatch, arr_to_ptcloud


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


def main(idx):
    pcm = PCMatch(*cvt_load_pcd(conf.dblist[idx]))

    print(len(pcm.rotmatrix))
    return

    MatlabEngine.start()
    fig = pyplot.figure()
    ax = Axes3D(fig)

    pcm.axis_match(1, 0)

    # scale = pcm.scale_match()
    # ax.scatter(*np.asarray(pcm.arrays[1]).T, s=1, marker='.', color='b')

    # pcm.rotmatrix()

    for epoch in range(1):
        for iter in range(3):
            scale = pcm.scale_match(coaxis=True)
            transf, fitness = pcm.icp_match()

        transf = pcm.icpf_match(registration='Affine')

    ptarray, pmarray = (np.asarray(a) for a in pcm.arrays)

    ax.scatter(*ptarray.T, s=1, marker='.', color='g')
    ax.scatter(*pmarray.T, s=1, marker='.', color='r')

    cch = pclsimilarity(arr_to_ptcloud(ptarray), arr_to_ptcloud(pmarray))
    print(f'{cch=}')

    pyplot.show()


if __name__ == '__main__':
    main(1)
