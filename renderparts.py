import os
from operator import itemgetter

import glfw
import numpy as np
from numpy.linalg import norm
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from cfgreader import conf
from clusterpartnet import BkThread, CLUSTER_DIM
from showobj import ShowObj


def triangle_area(a):
    a = np.square(a - np.roll(a, 1, axis=1))
    a = np.sqrt(np.sum(a, axis=2))
    p = np.sum(a, axis=1) / 2
    s = p * (p - a[:, 0]) * (p - a[:, 1]) * (p - a[:, 2])
    s[s < 0] = 0
    return np.sqrt(s)


def best_medoids(a):
    area_score = a[:, 3]
    area = np.sum(area_score)
    area_score = area_score / np.sum(area_score)
    a = a[:, :3]
    sim_score = area_score * np.abs(cosine_similarity(a)) @ area_score
    idx = np.argmax(sim_score)
    return area, a[idx]


def best_mean(a):
    area_score = a[:, 3]
    area = np.sum(area_score)
    area_score = area_score / np.sum(area_score)
    a = a[:, :3] * area_score[:, np.newaxis]

    return area, np.mean(a, axis=0)


def do_norm_calc(mtl_list, label_list):
    all_vertices = [list() for _ in range(CLUSTER_DIM)]
    for material, label in zip(mtl_list, label_list):
        la, c, p = np.asanyarray(label, dtype=np.int32).reshape([-1, 3]).T
        assert np.all(la == c) and np.all(la == p)
        a = np.asanyarray(material.vertices, dtype=np.float32).reshape([-1, 6])
        p = triangle_area(np.reshape(a[:, 3:], (-1, 3, 3)))
        a = np.reshape(normalize(a[:, :3]), (-1, 3, 3))
        a = np.concatenate((np.mean(a, axis=1), p[:, np.newaxis]), axis=1)
        for c, vertices in enumerate(all_vertices):
            vertices.append(a[la == c])

    ret = [((i, len(a),) + best_medoids(a)) for i, a in enumerate(map(np.concatenate, all_vertices))]
    ret.sort(key=itemgetter(2), reverse=True)
    return ret


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

    def __call__(self, *args, **kwargs):
        glfw.post_empty_event()

    def __init__(self, scene: Wavefront):
        self.bkt = BkThread(scene.mesh_list, self)
        self.cluster_cls = None
        self.cluster_id = None
        super().__init__(scene)

    def do_part(self, partid):
        mesh = self.scene.mesh_list[partid]
        mtl, = mesh.materials
        mtl_im_id, cls_name, file_name = conf.get_cls_from_mtlname(mtl.name)
        file_name, _ = os.path.splitext(file_name)
        mesh_name, _ = os.path.splitext(mesh.name)
        assert file_name == mesh_name
        print("{}({})".format(file_name, cls_name), end=' ')

        cls_name = conf.find_group_name(cls_name)
        if self.cluster_cls != cls_name:
            self.cluster_cls = cls_name
            self.cluster_id = 0
        elif self.cluster_id >= 2:
            self.cluster_id = 0
        else:
            self.cluster_id += 1
        print("> {}({}):".format(cls_name, self.cluster_id), end=' ')

        id_list, mtl_list = zip(*self.bkt.grp_dict[cls_name])
        label_list = [self.bkt.result_list[idx] for idx in id_list]
        vv = do_norm_calc(mtl_list, label_list)
        _, _, area, vv = vv[self.cluster_id]
        # vv = np.array([1, 1, 1], dtype=np.float32)
        vv /= norm(vv)
        self.initial_look_at = 4*vv
        vx, vy, vz = vv
        if vx != 0 or vz !=0:
            self.up_vector = np.array((0, 1, 0), dtype=np.float32)
        else:
            self.up_vector = np.array((1, 0, 0), dtype=np.float32)
        self.rot_angle[0] = 0
        self.rot_angle[1] = 0
        print(self.rot_angle)

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

    def show_obj(self):
        super().show_obj()
        self.bkt.can_exit = True
        self.bkt.join()


def main(idx):
    dblist = conf.dblist
    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ClsObj(scene)
        show.show_obj()
        if show.result == 1:
            idx = min(idx + 1, len(dblist) - 1)
        elif show.result == 2:
            idx = max(0, idx - 1)
        else:
            break


if __name__ == '__main__':
    main(0)
