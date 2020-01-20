import os
from threading import Lock

import glfw
import numpy as np
from imageio import imwrite
from numpy.linalg import norm
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light
from sklearn.metrics.pairwise import cosine_similarity

from cfgreader import conf
from clusterpartnet import BkThread, CLUSTER_DIM
from showobj import ShowObj


def triangle_area(a, pnorm=None):
    a = a - np.roll(a, 1, axis=1)
    if pnorm is not None:
        pnorm = pnorm / norm(pnorm)
        a = a - a * pnorm * pnorm
    a = np.sqrt(np.sum(np.square(a), axis=2))
    p = np.sum(a, axis=1) / 2
    a, b, c = a.T
    s = p * (p - a) * (p - b) * (p - c)
    s[s < 0] = 0
    return np.sqrt(s)


def best_medoids(p, area_ratio=0.5, sign_norm=None):
    a = p[:, :, :3]
    a = np.mean(a / norm(a, axis=2, keepdims=True), axis=1)
    area_score = triangle_area(p[:, :, 3:])
    total_area = np.sum(area_score)
    area_score = area_score / np.sum(area_score)
    num = len(area_score)
    if area_ratio < 1:
        idx = np.argsort(area_score * np.abs(cosine_similarity(a)) @ area_score)
        for num, area in enumerate(np.cumsum(area_score[idx])):
            if area >= area_ratio:
                break
        num += 1
    else:
        idx = np.arange(num)
    a = a[idx[:num]]
    if sign_norm is None:
        sign_norm = a[0]
    signs = np.sign(np.dot(a, sign_norm))
    a = a * signs[:, np.newaxis]
    area_score = area_score[idx[:num]]
    pnorm = np.sum(a * area_score[:, np.newaxis], axis=0) / np.sum(area_score)
    proj_area = np.sum(triangle_area(p[:, :, 3:], pnorm))
    return total_area, proj_area, pnorm


def do_norm_calc(mtl_list, label_list):
    all_vertices = [list() for _ in range(CLUSTER_DIM)]
    for material, label in zip(mtl_list, label_list):
        la, a, c = np.asanyarray(label, dtype=np.int32).reshape([-1, 3]).T
        assert np.all(la == a) and np.all(la == c)
        a = np.asanyarray(material.vertices, dtype=np.float32).reshape([-1, 3, 6])
        for c, vertices in enumerate(all_vertices):
            vertices.append(a[c == la])

    ret = [best_medoids(a) for a in map(np.concatenate, all_vertices)]

    for i in range(CLUSTER_DIM):
        pi, ni = (i - 1) % CLUSTER_DIM, (i + 1) % CLUSTER_DIM
        vertices = np.concatenate(all_vertices[pi] + all_vertices[ni])
        pn, nn = ret[pi][2], ret[ni][2]
        pn, nn = pn / norm(pn), nn / norm(nn)
        pn, nn = pn + nn, pn - nn
        aret, bret = (best_medoids(vertices, sign_norm=n) for n in (pn, nn))
        if aret[1] >= bret[1]:
            ret.append(aret)
        else:
            ret.append(bret)
        pass

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
    CLS_COLOR = ['R', 'G', 'B', 'GB', 'RB', 'RG']

    def __call__(self, cls_name: str, *args, **kwargs):
        if cls_name:
            id_list, mtl_list = zip(*self.bkt.grp_dict[cls_name])
            label_list = [self.bkt.result_list[idx] for idx in id_list]
            self.cluster_norm[cls_name] = do_norm_calc(mtl_list, label_list)

        if self.cluster_color:
            glfw.post_empty_event()
            return

        if not cls_name:
            self.render_lock.acquire()
            self.result = 1
            glfw.set_window_should_close(self.window, GL_TRUE)
            glfw.post_empty_event()
            return

        id_list, mtl_list = zip(*self.bkt.grp_dict[cls_name])

        self.render_lock.acquire()
        self.del_set = set(idx for idx in range(len(self.scene.mesh_list)) if idx not in set(id_list))
        self.sel_set = set()
        self.look_at_reset()
        self.render_name = cls_name.replace('/', '_')
        print(self.del_set)
        glfw.post_empty_event()

        for c in range(2 ** CLUSTER_DIM - 2):
            self.render_lock.acquire()
            self.look_at_cls(cls_name, c)
            self.render_name = cls_name.replace('/', '_') + '_look_{}+'.format(self.CLS_COLOR[c])
            glfw.post_empty_event()

            self.render_lock.acquire()
            self.initial_look_at = -self.initial_look_at
            self.render_name = cls_name.replace('/', '_') + '_look_{}-'.format(self.CLS_COLOR[c])
            glfw.post_empty_event()

    def __init__(self, dblist):
        self.dblist = dblist
        self.imageid = 0
        self.bkt = None
        self.cluster_cls = None
        self.cluster_id = None
        self.cluster_color = False
        self.cluster_norm = dict()
        self.render_lock = None
        self.render_name = None
        self.window = None
        super().__init__(self.load_image())

    def load_image(self):
        im_id = self.dblist[self.imageid]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        for material in scene.materials.values():
            material.ambient = [0.2, 0.2, 0.2, 1.0]
        self.bkt = BkThread(scene.mesh_list, self)
        return scene

    def look_at_reset(self):
        self.rot_angle = np.array((38.0, -17.0), dtype=np.float32)
        self.initial_look_at = np.array((0, 0, 3), dtype=np.float32)
        self.up_vector = np.array((0, 1, 0), dtype=np.float32)

    def look_at_cls(self, cls_name, cid=0):
        if cls_name not in self.cluster_norm:
            print("Not finished!")
            return
        vv = self.cluster_norm[cls_name]
        area, parea, vv = vv[cid]
        vv /= norm(vv)
        self.initial_look_at = 3 * vv
        vx, vy, vz = vv
        if vx != 0 or vz != 0:
            self.up_vector = np.array((0, 1, 0), dtype=np.float32)
        else:
            self.up_vector = np.array((1, 0, 0), dtype=np.float32)
        self.rot_angle[0] = 0
        self.rot_angle[1] = 0
        print("total_area {} projected_area {}".format(area, parea), vv)

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
        elif self.cluster_id >= 5:
            self.cluster_id = 0
        else:
            self.cluster_id += 1
        print("> {}({}):".format(cls_name, self.CLS_COLOR[self.cluster_id]), end=' ')
        self.look_at_cls(cls_name, self.cluster_id)

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

        if vertex_format == GL_C4F_N3F_V3F and self.cluster_color:
            glEnable(GL_COLOR_MATERIAL)

        glInterleavedArrays(vertex_format, 0, material.gl_floats)
        glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

        self.bkt.lock_list[idx].release()

        glPopAttrib()
        glPopClientAttrib()

    def window_load(self, window):
        self.render_lock = Lock()
        self.window = window
        self.result = 0
        if not self.cluster_color:
            self.render_lock.acquire()
            self.del_set = set()
            self.sel_set = set()
            self.render_name = 'render'
        self.look_at_reset()
        self.bkt.start()

    def window_closing(self, window):
        if self.result == 1:
            self.imageid = min(self.imageid + 1, len(self.dblist) - 1)
        elif self.result == 2:
            self.imageid = max(0, self.imageid - 1)
        else:
            return
        self.bkt.can_exit = True
        self.bkt.join()
        self.scene = self.load_image()
        glfw.set_window_should_close(window, GL_FALSE)
        self.window_load(window)

    def show_obj(self):
        super().show_obj()
        self.bkt.can_exit = True
        self.bkt.join()

    def draw_model(self):
        super().draw_model()
        if self.render_lock.locked():
            img = self.save_to_buffer()
            self.render_lock.release()
            im_id = self.bkt.im_id
            im_name = '{}.png'.format(self.render_name)
            file_path = os.path.join(conf.render_dir, im_id)
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            imwrite(os.path.join(file_path, im_name), img)


def main(idx):
    show = ClsObj(conf.dblist)
    show.show_obj()


if __name__ == '__main__':
    main(0)
