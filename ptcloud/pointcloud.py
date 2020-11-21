import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pcl

from ptcloud.pcmatch import PCMatch, pclsimilarity, arr2pt
from tools import blender_convert
from tools.blender_convert import download_id
from tools.cfgreader import conf

blender_convert.DATA_URL = conf.partnet_url


def cvt_obj2pcd(file, outdir, argv0='pcl_mesh_sampling', **kwargs) -> subprocess.CompletedProcess:
    if not os.path.exists(file):
        raise FileNotFoundError(file)
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
    return subprocess.run(argv, capture_output=True, text=True, check=True)


def cvt_load_pcd(im_id, faceid=3, after_merging=True, n_samples=10000):
    p_dir = os.path.join(conf.pcd_dir, im_id)
    cvt_obj2pcd(os.path.join(conf.data_dir, im_id + '.obj'), p_dir,
                n_samples=n_samples, leaf_size=0.001)

    rname = f'parts_render_after_merging_0.face{faceid}'
    if after_merging:
        try:
            cvt_obj2pcd(os.path.join(conf.pix2mesh_dir, im_id, rname + '.obj'), p_dir,
                        n_samples=n_samples, leaf_size=0.0005)
        except (FileNotFoundError, subprocess.CalledProcessError):
            after_merging = False

    if not after_merging:
        rname = f'parts_render_0.face{faceid}'
        cvt_obj2pcd(os.path.join(conf.pix2mesh_dir, im_id, rname + '.obj'), p_dir,
                    n_samples=n_samples, leaf_size=0.0005)

    ptcloud = pcl.load(os.path.join(p_dir, im_id + '.pcd'))
    pmcloud = pcl.load(os.path.join(p_dir, rname + '.pcd'))

    print(f'{ptcloud.size=}, {pmcloud.size=}')
    ptcloud, pmcloud = np.asarray(ptcloud), np.asarray(pmcloud)
    return ptcloud, pmcloud


def draw_bbox(ax, corner_points):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.scatter(*corner_points.T, s=2, marker='+', color='b')
    vertices = ((0, 1, 2, 3, 0, 4, 5, 6, 7, 4),
                (1, 5), (2, 6), (3, 7))
    poly3d = tuple(tuple(corner_points[iy] for iy in ix) for ix in vertices)
    line = Line3DCollection(poly3d, colors='k', linewidths=0.2, linestyles=':')
    ax.add_collection3d(line)


def group_parts_pcds(im_id, n_samples=5000):
    p_dir = os.path.join(conf.pcd_dir, im_id)
    group_dict = {}
    for name, f in download_id(im_id):
        print(name, f)
        cvt_obj2pcd(f, p_dir, n_samples=n_samples, leaf_size=0.001)
        fn, ext = os.path.splitext(os.path.basename(f))
        group_name = conf.find_group_name(name)
        points = np.asarray(pcl.load(os.path.join(p_dir, fn + '.pcd')))
        if group_name in group_dict:
            points = np.concatenate((group_dict[group_name], points))
        group_dict[group_name] = points
    return group_dict


def filter_region(ref, a, margin=0.05):
    min_a, max_a = np.min(ref, axis=0), np.max(ref, axis=0)
    ind = np.all(np.logical_and(min_a - margin < a, a < max_a + margin), axis=1)
    return a[ind, :]


def eval_id(im_id, draw_plot=True, log_file=sys.stdout):
    from tools.matlabengine import MatEng
    pcm = PCMatch(*cvt_load_pcd(im_id))

    MatEng.start(count=2)

    align1, align0 = pcm.axis_match(1, 0)
    sim_score, rot_trans, offset = pcm.rotmatrix_match()

    print('Rot Sim Score:', im_id, sim_score, file=log_file)

    # draw_bbox(ax, (align1.corner_points - align1.mean) @ (align1.components.T @ rot_trans.T) + offset)
    # ax.scatter(*np.asarray(pcm.arrays[1]).T, s=1, marker='.', color='b')

    reg_score = pcm.register()

    print('Reg Sim Score:', im_id, reg_score, file=log_file)

    ptarray, pmarray = (np.asarray(a) for a in pcm.arrays)

    group_score = {}
    for name, pcd in group_parts_pcds(im_id).items():
        ref = align0.transform(pcd)
        a = filter_region(ref, pmarray)
        if a.size:
            group_score[name] = pclsimilarity(arr2pt(ref), arr2pt(a))
        else:
            group_score[name] = np.Inf

    if draw_plot:
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        ax.scatter(*ptarray.T, s=1, marker='.', color='g')
        ax.scatter(*pmarray.T, s=1, marker='.', color='r')

        for axis_name in 'xyz':
            getattr(ax, 'set_%slim3d' % axis_name)(-1, 1)

        pyplot.show()

    for cls, score in group_score.items():
        print('Score:', cls, score, file=log_file, flush=True)

    results = SimpleNamespace(im_id=im_id, align1=align1, align0=align0,
                              sim_score=sim_score, rot_trans=rot_trans, offset=offset,
                              reg_score=reg_score, arrays=pcm.arrays, group_score=group_score)
    return vars(results)


def main():
    with open('../pointcloud.log', 'w') as log_file:
        for im_id in conf.dblist:
            try:
                results = eval_id(im_id, draw_plot=False, log_file=log_file)
                p_dir = os.path.join(conf.pcd_dir, im_id)
                cache_file = os.path.join(p_dir, 'result.pkl')
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except:
                e = sys.exc_info()[0]
                print("Error: %s" % e, file=log_file, flush=True)


if __name__ == '__main__':
    main()
