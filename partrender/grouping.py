import os
from typing import Final

import glfw
import numpy as np
from numpy.linalg import norm
from pyglet.gl import *
from sklearn.metrics.pairwise import cosine_similarity

from partrender.clustering import BkThread, CLUSTER_DIM
from partrender.rendering import RenderObj
from tools.cfgreader import conf


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


class GroupObj(RenderObj):
    CLS_COLOR: Final = ['R', 'G', 'B', 'GB', 'RB', 'RG']

    def __init__(self, start_id, auto_generate=False):
        self.bkt = None
        self.cluster_cls = None
        self.cluster_id = None
        self.cluster_norm = dict()
        super(GroupObj, self).__init__(start_id, auto_generate)

    def load_image(self):
        scene = super(GroupObj, self).load_image()
        self.bkt = BkThread(scene.mesh_list, self.lock_list, self)
        return scene

    def __call__(self, cls_name: str, *args, **kwargs):
        if cls_name:
            id_list, mtl_list = zip(*self.bkt.grp_dict[cls_name])
            label_list = [self.bkt.result_list[idx] for idx in id_list]
            self.cluster_norm[cls_name] = do_norm_calc(mtl_list, label_list)

        if self.cluster_color:
            glfw.post_empty_event()
            return

        if not cls_name:
            self.result = 1
            glfw.set_window_should_close(self.window, GL_TRUE)
            glfw.post_empty_event()
            return

        id_list, mtl_list = zip(*self.bkt.grp_dict[cls_name])

        with self.set_render_name(cls_name.replace('/', '_')):
            self.del_set = set(idx for idx in range(len(self.scene.mesh_list)) if idx not in set(id_list))
            self.sel_set = set()
            self.look_at_reset()
            print(self.del_set)

        for c in range(2 ** CLUSTER_DIM - 2):
            with self.set_render_name(cls_name.replace('/', '_') + '_look_{}+'.format(self.CLS_COLOR[c])):
                self.look_at_cls(cls_name, c)

            with self.set_render_name(cls_name.replace('/', '_') + '_look_{}-'.format(self.CLS_COLOR[c])):
                self.initial_look_at = -self.initial_look_at

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

    def window_load(self, window):
        self.cluster_cls = None
        self.cluster_id = None
        self.cluster_norm = dict()
        super(GroupObj, self).window_load(window)
        self.bkt.start()

    def window_closing(self, window):
        self.bkt.exit_wait()
        super(GroupObj, self).window_closing(window)


def main(idx, autogen=True):
    show = GroupObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
