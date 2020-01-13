import os
import time

from pywavefront import Wavefront
from pywavefront.mesh import Mesh
from pywavefront.material import Material
from cfgreader import conf
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import spectral_embedding
from sklearn.cluster import k_means
from showobj import ShowObj
from threading import Thread, Lock
from pyglet.gl import *
from pywavefront.visualization import gl_light
from collections import defaultdict


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def get_normal_cls(vertices):
    a = np.array(vertices, dtype=np.float32).reshape([-1, 6])
    nvtx, _ = a.shape
    print("Vtx Count: %d" % nvtx)

    n9f = np.reshape(normalize(a[:, :3]), (-1, 3, 3))
    n3f = np.mean(n9f, axis=1)

    with Timer("abs_cosine_similarity"):
        sim_mat = np.abs(cosine_similarity(n3f))
    with Timer("spectral_embedding"):
        embedding = spectral_embedding(sim_mat, n_components=3, drop_first=False)
    with Timer("k_means"):
        centers, labels, _ = k_means(embedding, n_clusters=3)

    ret = [np.count_nonzero(labels == i) for i in range(3)]
    max_lbl_id = np.argmax(ret)
    ret.sort(reverse=True)
    print('\t'.join(str(s) for s in ret))
    print(centers)
    # plt.figure()
    # plt.scatter(embedding[:, 1], embedding[:, 2], c=labels)
    # plt.show()

    dist = euclidean_distances(embedding, centers)
    min_norm_ids = np.argmin(dist, axis=0)
    best_norm_id = min_norm_ids[max_lbl_id]

    labels = np.repeat(labels, 3)
    return a, labels, embedding, n3f[best_norm_id]


def normal_cls_functor(pixel_format):
    pixel_normal = {
        'N3F_V3F': get_normal_cls
    }
    return pixel_normal[pixel_format]


class BkThread(Thread):
    def __init__(self, mesh_list, callback=None):
        self.grp_dict = defaultdict(list)
        self.mesh_list = mesh_list
        self.lock_list = [Lock() for i in range(len(mesh_list))]
        self.result_list = [None for i in range(len(mesh_list))]
        self.can_exit = False
        self.callback = callback
        super().__init__(daemon=True)

    def do_cluster(self, mtl_id_list):
        _, mtl = mtl_id_list[0]
        vertices = mtl.vertices
        vertex_format = mtl.vertex_format
        for _, material in mtl_id_list[1:]:
            assert vertex_format == material.vertex_format
            vertices += material.vertices
        return normal_cls_functor(vertex_format)(vertices)

    def change_mtl(self, idx, material: Material, a, labels):
        nvtx, _ = a.shape

        self.lock_list[idx].acquire(blocking=True)

        color = np.zeros(shape=(nvtx, 4), dtype=np.float32)
        for lbl, arr in zip(labels, color):
            arr[lbl] = 1

        material.gl_floats = np.concatenate((color, a), axis=1).ctypes
        material.triangle_count = nvtx
        material.vertex_format = 'C4F_N3F_V3F'

        self.lock_list[idx].release()

        self.result_list[idx] = norm

    def add_to_dict(self):
        for idx, mesh in enumerate(self.mesh_list):
            print('analysis mesh: %s' % mesh.name)
            for mtl in mesh.materials:
                cls_name, file_name = conf.mtl_name_to_cls(mtl.name)
                file_name, _ = os.path.splitext(file_name)
                mesh_name, _ = os.path.splitext(mesh.name)
                assert mesh_name == file_name
                cls_name = str(cls_name).strip()
                while cls_name:
                    if cls_name in conf.groupset:
                        self.grp_dict[cls_name].append((idx, mtl))
                        print('Adding to ' + cls_name)
                        break
                    strindex = cls_name.rindex('/')
                    cls_name = cls_name[:strindex]

    def run(self):
        self.add_to_dict()

        if self.can_exit:
            return

        for cls_name, mtl_id_list in self.grp_dict.items():
            a, labels, embedding, norm = self.do_cluster(mtl_id_list)
            for idx, mtl in mtl_id_list:
                self.change_mtl(idx, mtl, a, labels)
            if callable(self.callback):
                self.callback(cls_name=cls_name,
                              idlist=np.fromiter((idx for idx, mtl in mtl_id_list),
                                                 dtype=np.int, count=len(mtl_id_list)),
                              a=a, labels=labels, embedding=embedding, norm=norm)
            if self.can_exit:
                return


class ClsObj(ShowObj):
    VERTEX_FORMATS = {
        'V3F': GL_V3F,
        'C3F_V3F': GL_C3F_V3F,
        'N3F_V3F': GL_N3F_V3F,
        'T2F_V3F': GL_T2F_V3F,
        'C4F_N3F_V3F': GL_C4F_N3F_V3F,
        'T2F_C3F_V3F': GL_T2F_C3F_V3F,
        'T2F_N3F_V3F': GL_T2F_N3F_V3F,
        'T2F_C4F_N3F_V3F': GL_T2F_C4F_N3F_V3F,
    }

    def __init__(self, scene: Wavefront):
        self.bkt = BkThread(scene.mesh_list)
        super().__init__(scene)

    def do_part(self, partid):
        mesh = self.scene.mesh_list[partid]
        print('\t'.join("%s(%s)" % (mtl.name, mtl.vertex_format) for mtl in mesh.materials))

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        """Draw a single material"""
        self.bkt.lock_list[idx].acquire(blocking=True)

        if material.gl_floats is None:
            material.gl_floats = np.asarray(material.vertices, dtype=np.float32).ctypes
            material.triangle_count = len(material.vertices) / material.vertex_size

        vertex_format = self.VERTEX_FORMATS.get(material.vertex_format)
        if not vertex_format:
            raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)

        glDisable(GL_TEXTURE_2D)
        glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
        glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
        glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
        glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
        glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))
        glEnable(GL_LIGHT0)

        if material.has_normals:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)

        if vertex_format == GL_C4F_N3F_V3F:
            glEnable(GL_COLOR_MATERIAL)

        glInterleavedArrays(vertex_format, 0, material.gl_floats)
        glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

        self.bkt.lock_list[idx].release()

        glPopAttrib()
        glPopClientAttrib()

    def window_load(self, window):
        self.bkt.start()


def save_data(im_id: str):
    print('Init saving: %s' % im_id)

    def do_save(*args, cls_name: str, **kwargs):
        save_dir = os.path.join(conf.render_dir, im_id)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, '{}.npz'.format(cls_name.replace('/', '-')))
        np.savez(save_file, *args, **kwargs)

    return do_save


def main(idx):
    dblist = conf.dblist
    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        bkt = BkThread(scene.mesh_list, save_data(im_id))
        bkt.start()
        idx = min(idx + 1, len(dblist) - 1)
        bkt.join()


if __name__ == '__main__':
    main(50)
