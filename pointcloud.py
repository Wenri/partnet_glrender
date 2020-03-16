import os

import numpy as np
import pcl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cfgreader import conf
from matlabengine import MatEng
from pcmatch import PCMatch


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


def cvt_load_pcd(fname, faceid=3, after_merging=True, n_samples=10000):
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

    MatEng.start(count=2)
    fig = pyplot.figure()
    ax = Axes3D(fig)

    pcm.center_match()
    pcm.scale_match(coaxis=False)
    pmax, ptax = pcm.axis_match(1, 0)
    # pcm.scale_match(coaxis=False)
    # pcm.rotmatrix_match()
    #
    # ax.scatter(*np.asarray(pcm.arrays[1]).T, s=1, marker='.', color='b')
    #
    # for iter in range(3):
    #     pcm.scale_match(coaxis=False)
    #     pcm.icp_match()
    #
    # for iter in range(3):
    #     pcm.scale_match(coaxis=False)
    #     pcm.icpf_match(registration='Affine')

    ptarray, pmarray = (np.asarray(a) for a in pcm.arrays)

    ax.scatter(*ptarray.T, s=1, marker='.', color='g')
    ax.scatter(*pmarray.T, s=1, marker='.', color='r')

    for axis_name in 'xyz':
        getattr(ax, 'set_%slim3d' % axis_name)(-1, 1)

    pyplot.show()


if __name__ == '__main__':
    main(0)
