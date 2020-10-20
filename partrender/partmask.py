import faulthandler
import hashlib
import operator
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from functools import reduce, partial
from itertools import chain
from math import cos, sin, pi
from pathlib import Path
from threading import Thread

import numpy as np
import pcl
from more_itertools import first
from pyglet.gl import *
from pywavefront.material import Material

from partrender.rendering import RenderObj
from ptcloud.pcmatch import PCMatch
from ptcloud.pointcloud import cvt_obj2pcd
from tools.blender_convert import ShapenetFileHelper
from tools.blender_convert import load_obj_files, load_json
from tools.cfgreader import conf


# The coefficients were taken from OpenCV https://github.com/opencv/opencv
# I'm not sure if the values should be clipped, in my (limited) testing it looks alright
#   but don't hesitate to add rgb.clip(0, 1, rgb) & yuv.clip(0, 1, yuv)
#
# Input for these functions is a numpy array with shape (height, width, 3)
# Change '+= 0.5' to '+= 127.5' & '-= 0.5' to '-= 127.5' for values in range [0, 255]

def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.147108, 0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400, 0.435912, -0.099978]
    ])
    yuv = np.dot(rgb, m)
    yuv[..., 1:] += 0.5
    return yuv


def yuv2rgb(yuv):
    m = np.array([
        [1.000, 1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    yuv[..., 1:] -= 0.5
    rgb = np.dot(yuv, m)
    return rgb


class MaskObj(RenderObj):
    def __init__(self, start_id, auto_generate=False, load_shapenet=True):
        super(MaskObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)
        self.n_lights = 8
        self.n_samples = 10
        self.matched_matrix = None
        self.should_apply_trans = False
        self.should_load_shapenet = load_shapenet
        self.old_scene = None
        self.model_id = 0
        self.obj_ins_map = None

        self.act_key('T', self.swap_scene)
        self.act_key('G', partial(self.swap_scene, toggle_trans=False))

        if auto_generate:
            conf.save_groupset(os.path.join(conf.partmask_dir, 'grouping.txt'))

    def swap_scene(self, toggle_trans=True):
        if self.old_scene:
            self.old_scene = self.update_scene(self.old_scene)
            if toggle_trans:
                self.should_apply_trans = not self.should_apply_trans

    def load_pcd(self, base_dir, n_samples=20000, leaf_size=0.001):
        im_id = conf.dblist[self.imageid]
        imfile = os.path.join(base_dir, "{}.obj".format(im_id))
        with tempfile.TemporaryDirectory() as tempdirname:
            ret = cvt_obj2pcd(imfile, tempdirname, n_samples=n_samples, leaf_size=leaf_size)
            if ret.stderr:
                print(f'cvt_obj2pcd {os.path.basename(base_dir)} {im_id}: ', ret.stderr, file=sys.stderr)
            return pcl.load(os.path.join(tempdirname, '{}.pcd'.format(im_id)))

    def calc_apply_matched_matrix(self):
        partnet_pcd = self.load_pcd(conf.data_dir, leaf_size=0.002)
        shapenet_pcd = self.load_pcd(conf.shapenet_dir, leaf_size=0.001)
        pcm = PCMatch(partnet_pcd, shapenet_pcd)
        sim_min, trans, offset = pcm.rotmatrix_match()
        trans_m = np.hstack((np.transpose(trans), offset[:, np.newaxis]))
        self.matched_matrix = np.vstack((trans_m, np.array([[0., 0., 0., 1.]])))
        self.should_apply_trans = True

    def load_shapenet(self):
        helper = ShapenetFileHelper(prefix=conf.shapenet_prefix)
        search_path = first(helper.find_model_id(conf.dblist[self.imageid]))
        print(f"{search_path=}")
        self.model_id = os.path.basename(search_path)
        scene = self.load_image(conf.shapenet_dir)
        for m in chain.from_iterable(mesh.materials for mesh in scene.mesh_list):
            if t := m.texture:
                t._search_path = Path(search_path)
        return scene

    def window_load(self, window):
        super(MaskObj, self).window_load(window)

        self.old_scene = None
        if self.should_load_shapenet:
            try:
                scene = self.load_shapenet()
                self.calc_apply_matched_matrix()
                self.old_scene = self.update_scene(scene)
            except subprocess.CalledProcessError as e:
                print(first(e.cmd), 'err code', e.returncode, file=sys.stderr)
            except (FileNotFoundError, IOError) as e:
                print(e, file=sys.stderr)
        self.collect_instance_id()

        Thread(target=self, daemon=True).start()

    def draw_model(self):
        with ExitStack() as stack:
            if self.should_apply_trans:
                stack.enter_context(self.matrix_trans(self.matched_matrix))
            super(MaskObj, self).draw_model()

    def random_seed(self, s, seed=0xdeadbeef):
        # seeding numpy random state
        halg = hashlib.sha1()
        print(s, end='/')
        s = 'Random seed {} with {} lights'.format(s, self.n_lights)
        halg.update(s.encode())
        s = halg.digest()
        s = reduce(operator.xor, (int.from_bytes(s[i * 4:i * 4 + 4], byteorder='little') for i in range(len(s) // 4)))
        s ^= seed
        rs = np.random.RandomState(seed=s)
        print(f'{rs.random():.4f}', end=' ')

        # random view angle
        rx, ry = rs.random_sample(size=2)
        self.rot_angle = np.array((80 * 2 * (rx - 0.5), -45.0 * ry), dtype=np.float32)

        # random light color
        def rand_color(power=1.0, color_u=0.5, color_v=0.5):
            base_color = yuv2rgb(np.array([power, color_u, color_v]))
            base_color = rgb2yuv(np.clip(base_color, 0, 1))
            color = rs.standard_normal(size=3)
            color *= np.array([0.01, 0.05, 0.05])
            color += base_color
            r, g, b = np.clip(yuv2rgb(color), 0, 1, dtype=np.float32)
            return r, g, b, 1.0

        def rand_pos(*pos):
            pos_sample = rs.standard_normal(size=3) / 3
            x, y, z = pos_sample + np.array(pos)
            return x, y, z, 0.0

        # random light source
        self.clear_light_source()
        w, d, s = 4, 1, pi / (self.n_lights - 1)
        for i in range(self.n_lights):
            self.add_light_source(ambient=rand_color(0.2 / self.n_lights),
                                  diffuse=rand_color(0.8 / self.n_lights),
                                  specular=rand_color(0.8 / self.n_lights),
                                  position=rand_pos(w * cos(s * i), 4, 4 - d * sin(s * i)))

        # random vertex color
        u, v = 0.6 * rs.random_sample(size=2) + 0.2
        diffuse = yuv2rgb(np.array([0.5, u, v]))
        diffuse = rgb2yuv(np.clip(diffuse, 0, 1))

        def change_mtl(idx, material: Material):
            a = np.array(material.vertices, dtype=np.float32).reshape([-1, 6])
            n_vtx, _ = a.shape
            with self.lock_list[idx]:
                color = rs.standard_normal(size=(n_vtx, 3))
                alpha = np.ones(shape=(n_vtx, 1), dtype=np.float32)
                color *= np.array([0.01, 0.05, 0.05])
                color += diffuse
                color = np.clip(yuv2rgb(color), 0, 1, dtype=np.float32)
                material.gl_floats = np.concatenate((color, alpha, a), axis=1).ctypes
                material.triangle_count = n_vtx
                material.vertex_format = 'C4F_N3F_V3F'

        for i, mesh in enumerate(self.scene.mesh_list):
            for m in mesh.materials:
                change_mtl(i, m)

    def convert_mesh(self, mesh_list):
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, conf.dblist[self.imageid] + '.obj')
            with open(filename, mode='w') as f:
                print("# OBJ file", file=f)
                for v in self.scene.vertices:
                    print("v %.4f %.4f %.4f" % v[:], file=f)
                for m in mesh_list:
                    for p in m.faces:
                        print("f %d %d %d" % tuple(i + 1 for i in p), file=f)
            return np.array(self.load_pcd(d, leaf_size=0.001))

    def collect_instance_id(self):
        meta = load_json(conf.dblist[self.imageid])

        rlookup = {os.path.splitext(mesh.name)[0]: idx + 1 for idx, mesh in enumerate(self.scene.mesh_list)}
        self.obj_ins_map = defaultdict(list)
        for ins_path, cls_name, obj in load_obj_files(meta):
            obj_name = os.path.splitext(os.path.basename(obj))[0]
            try:
                ins_id = conf.trim_ins_path(ins_path.split('/'), cls_name)
                self.obj_ins_map[ins_id].append(rlookup[obj_name])
            except ValueError as e:
                print('Skipping {} due to {}'.format(obj_name, e), file=sys.stderr)
                self.del_set.add(rlookup[obj_name] - 1)

    def __call__(self, *args, **kwargs):
        try:
            im_id = conf.dblist[self.imageid]
            print('rendering:', self.imageid, im_id, self.model_id, end=' ')

            if not self.view_mode:
                save_dir = os.path.join(self.render_dir, im_id)
                os.makedirs(save_dir, exist_ok=True)
                scene = self.old_scene if self.old_scene else self.scene
                with open(os.path.join(save_dir, 'render-CLSNAME.txt'), mode='w') as f:
                    for idx, mesh in enumerate(scene.mesh_list):
                        for material in mesh.materials:
                            conf_im_id, cls_name, file_name = conf.get_cls_from_mtlname(im_id, material.name)
                            assert conf_im_id == im_id
                            conf_mesh_name, _ = os.path.splitext(file_name)
                            mesh_name, _ = os.path.splitext(mesh.name)
                            assert conf_mesh_name == mesh_name
                            print(conf_im_id, idx, cls_name, file_name, file=f)
                ins_list = list(self.obj_ins_map.items())
                with open(os.path.join(save_dir, 'render-INSNAME.txt'), mode='w') as f:
                    for ins_path, meshes in ins_list:
                        print(ins_path, ','.join(map(str, meshes)), file=f)
                ins_pc = [self.convert_mesh([self.scene.mesh_list[idx - 1] for idx in meshes])
                          for _, meshes in ins_list]
                np.save(os.path.join(save_dir, 'render-INS_PC.npy'), ins_pc)
                for i in range(self.n_samples):
                    with self.set_render_name('seed_{}'.format(i), wait=True):
                        self.swap_scene()
                        self.random_seed('{}-{}'.format(self.imageid, i))
                    if self.old_scene:
                        with self.set_render_name('seed_{}T'.format(i), wait=True):
                            self.swap_scene()

                print('Switching...')
                self.set_fast_switching()
            else:
                self.render_ack.wait()
                print('Done.')

        except RuntimeError:
            return


def main(idx, autogen=True):
    faulthandler.enable()
    show = MaskObj(idx, autogen, load_shapenet=False)
    show.show_obj()


if __name__ == '__main__':
    main(0, autogen=True)
