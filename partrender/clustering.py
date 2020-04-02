import operator
import os
import time
from collections import defaultdict
from threading import Thread

import numpy as np
from numpy.linalg import norm
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.material import Material
from sklearn.cluster import k_means
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import cosine_similarity

from partrender.rendering import RenderObj
from tools.cfgreader import conf

CLUSTER_DIM = 3


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def get_normal_cls(a, max_samples=60000):
    a = np.asanyarray(a, dtype=np.float32).reshape([-1, 3, 6])
    a = a[:, :, :3]
    a = np.mean(a / norm(a, axis=2, keepdims=True), axis=1)

    n_samples, _ = a.shape
    print("Sample Count: %d" % n_samples)
    perm = np.random.permutation(n_samples)

    if n_samples > max_samples:
        print("Sampling to: %d" % max_samples)
        perm = perm[:max_samples]
    with Timer("abs_cosine_similarity"):
        sim_mat = np.abs(cosine_similarity(a[perm]))
    with Timer("spectral_embedding"):
        embedding = spectral_embedding(sim_mat, n_components=CLUSTER_DIM, drop_first=False)
    with Timer("k_means"):
        centers, labels, _ = k_means(embedding, n_clusters=CLUSTER_DIM)

    ret = [np.count_nonzero(labels == i) for i in range(3)]
    max_lbl_id = np.argmax(ret)
    ret.sort(reverse=True)
    print('\t'.join(str(s) for s in ret))
    # print(centers)
    # plt.figure()
    # plt.scatter(embedding[:, 1], embedding[:, 2], c=labels)
    # plt.show()

    # dist = euclidean_distances(embedding, centers)
    # min_norm_ids = np.argmin(dist, axis=0)
    # best_norm_id = min_norm_ids[max_lbl_id]

    full_labels = np.full(n_samples, -1)
    full_labels[perm] = labels
    full_labels = np.repeat(full_labels, 3)
    return perm, full_labels


def normal_cls_functor(pixel_format):
    pixel_normal = {
        'N3F_V3F': get_normal_cls
    }
    return pixel_normal[pixel_format]


def do_cluster(mtl_list):
    mtl = mtl_list[0]
    vertices = np.asanyarray(mtl.vertices, dtype=np.float32)
    vertex_format = mtl.vertex_format
    for material in mtl_list[1:]:
        assert vertex_format == material.vertex_format
        vertices1 = np.asanyarray(material.vertices, dtype=np.float32)
        vertices = np.concatenate((vertices, vertices1), axis=None)
    return normal_cls_functor(vertex_format)(vertices)


def load_cluster_dict(im_id, cls_name):
    print('loading {}: {} ...'.format(im_id, cls_name))
    file_path = os.path.join(conf.render_dir, im_id, '{}.npz'.format(cls_name.replace('/', '-')))
    if not os.path.exists(file_path):
        return None
    return np.load(file_path)


def save_cluster_dict(*args, im_id: str, cls_name: str, **kwargs):
    print('saving {}: {} ...'.format(im_id, cls_name), end=' ')
    save_dir = os.path.join(conf.render_dir, im_id)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, '{}.npz'.format(cls_name.replace('/', '-')))
    np.savez(save_file, *args, **kwargs)
    print('done.', flush=True)


class BkThread(Thread):
    def __init__(self, mesh_list, lock_list, callback=None):
        self.grp_dict = defaultdict(list)
        self.mesh_list = mesh_list
        self.lock_list = lock_list
        self.result_list = [None] * len(mesh_list)
        self.can_exit = False
        self.callback = callback
        self.im_id = None
        self.add_to_dict()
        super().__init__(daemon=True)

    def change_mtl(self, idx, material: Material, labels):
        a = np.array(material.vertices, dtype=np.float32).reshape([-1, 6])
        n_vtx, _ = a.shape
        assert n_vtx == len(labels)

        with self.lock_list[idx]:
            color = np.zeros(shape=(n_vtx, 4), dtype=np.float32)
            for lbl, arr in zip(labels, color):
                if lbl >= 0:
                    arr[lbl] = 1

            material.gl_floats = np.concatenate((color, a), axis=1).ctypes
            material.triangle_count = n_vtx
            material.vertex_format = 'C4F_N3F_V3F'

        self.result_list[idx] = labels

    def add_to_dict(self):
        for idx, mesh in enumerate(self.mesh_list):
            print('analysis mesh: %s' % mesh.name)
            mtl, = mesh.materials
            mtl_im_id, cls_name, file_name = conf.get_cls_from_mtlname(mtl.name)
            if self.im_id:
                assert mtl_im_id == self.im_id
            else:
                self.im_id = mtl_im_id
            file_name, _ = os.path.splitext(file_name)
            mesh_name, _ = os.path.splitext(mesh.name)
            assert mesh_name == file_name
            cls_name = conf.find_group_name(cls_name)
            self.grp_dict[cls_name].append((idx, mtl))
            print('Adding to ' + cls_name)

    def gen_cluster_result(self, cls_name: str, mtl_id_list: list):
        cluster_dict = load_cluster_dict(self.im_id, cls_name)
        id_list, mtl_list = zip(*mtl_id_list)
        if not cluster_dict:
            perm, labels = do_cluster(mtl_list)
            save_cluster_dict(im_id=self.im_id, cls_name=cls_name,
                              idlist=np.fromiter(id_list, dtype=np.int, count=len(id_list)),
                              perm=perm, labels=labels)
        else:
            idlist = cluster_dict['idlist']
            perm = cluster_dict['perm']
            labels = cluster_dict['labels']
            assert len(idlist) == len(id_list) and all(map(operator.eq, idlist, id_list))
        return perm, labels

    def run(self):
        if self.can_exit:
            return
        for cls_name, mtl_id_list in self.grp_dict.items():
            _, labels = self.gen_cluster_result(cls_name, mtl_id_list)
            if self.can_exit:
                return
            for idx, mtl in mtl_id_list:
                n_vertex = int(len(mtl.vertices) / 6)
                self.change_mtl(idx, mtl, labels[:n_vertex])
                labels = labels[n_vertex:]
            if self.can_exit or (callable(self.callback) and not self.callback(cls_name)):
                return
        self.callback(None)

    def async_exit(self):
        self.can_exit = True
        return self


def main(nskip):
    dblist = conf.dblist
    idx = len(dblist) - 1 - nskip
    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        bkt = BkThread(scene.mesh_list)
        bkt.start()
        idx = max(idx - 1, 0)
        bkt.join()
        if idx <= 0:
            break


if __name__ == '__main__':
    main(0)
