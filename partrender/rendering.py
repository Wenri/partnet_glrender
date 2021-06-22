import hashlib
import logging
import operator
import os
import sys
from contextlib import contextmanager, ExitStack, suppress
from ctypes import POINTER
from functools import partial, reduce
from itertools import chain
from math import cos, sin, pi
from threading import Lock, Condition, Event
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light, bind_texture

from partrender.showobj import ShowObj, get_gl_matrix
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


class RenderObj(ShowObj):
    _VERTEX_FORMATS: Final = {
        'V3F': GL_V3F,
        'C3F_V3F': GL_C3F_V3F,
        'N3F_V3F': GL_N3F_V3F,
        'T2F_V3F': GL_T2F_V3F,
        'C4F_N3F_V3F': GL_C4F_N3F_V3F,
        'T2F_C3F_V3F': GL_T2F_C3F_V3F,
        'T2F_N3F_V3F': GL_T2F_N3F_V3F,
        'T2F_C4F_N3F_V3F': GL_T2F_C4F_N3F_V3F,
    }

    def __init__(self, start_id, view_mode=True, render_dir=conf.render_dir):
        self.imageid = start_id
        self.view_mode = view_mode
        self.n_lights = 8
        self.render_lock = None
        self.render_cmd = None
        self.render_req = 0
        self.render_name = None
        self.render_ack = None
        self.render_dir = render_dir
        self.lock_list = None
        self.window = None

        logging.getLogger("pywavefront").addFilter(
            lambda r: 0 if r.msg.startswith("Unimplemented OBJ format statement 's' on line ") else 1
        )
        super(RenderObj, self).__init__(self.load_image(conf.data_dir))
        self.act_key('R', lambda w: self.save_buffer())

    def load_image(self, base_dir):
        im_id = conf.dblist[self.imageid]
        im_file = os.path.join(base_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file, collect_faces=True)

        for material in scene.materials.values():
            material.ambient = [0.2, 0.2, 0.2, 1.0]
        return scene

    def update_scene(self, scene):
        old_scene = super(RenderObj, self).update_scene(scene)
        self.lock_list = [Lock() for _ in range(len(scene.mesh_list))]
        return old_scene

    def random_seed(self, s, seed=0xdeadbeef, rotlr=(-180.0, 180.0), rotud=(-55.0, 0.0)):
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
        self.rot_angle = rs.uniform(*zip(rotlr, rotud))

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
            self.add_light_source(
                ambient=rand_color(0.2 / self.n_lights),
                diffuse=rand_color(0.8 / self.n_lights),
                specular=rand_color(0.8 / self.n_lights),
                position=rand_pos(w * cos(s * i), 4, 4 - d * sin(s * i))
            )

        # random vertex color
        u, v = 0.6 * rs.random_sample(size=2) + 0.2
        diffuse = yuv2rgb(np.array([0.5, u, v]))
        diffuse = rgb2yuv(np.clip(diffuse, 0, 1))

        def change_mtl(idx, material):
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

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        """Draw a single material"""

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        glDisable(GL_CULL_FACE)

        with ExitStack() as stack:
            stack.enter_context(self.lock_list[idx])
            if material.gl_floats is None:
                material.gl_floats = np.asarray(material.vertices, dtype=np.float32).ctypes
                material.triangle_count = len(material.vertices) / material.vertex_size

            vertex_format = self._VERTEX_FORMATS.get(material.vertex_format)
            if not vertex_format:
                raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

            if textures_enabled:
                # Fall back to ambient texture if no diffuse
                texture = material.texture or material.texture_ambient
                if texture and material.has_uvs:
                    try:
                        bind_texture(texture)
                        stack.callback(partial(glBindTexture, texture.image.target, 0))
                    except (IsADirectoryError, FileNotFoundError, IOError) as e:
                        print(f'Texture file missing: {e}', file=sys.stderr)
                        textures_enabled = False
                else:
                    textures_enabled = False
            if not textures_enabled:
                glDisable(GL_TEXTURE_2D)

            glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
            glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
            glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
            glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
            glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))

            if material.has_normals:
                glEnable(GL_LIGHTING)
            else:
                glDisable(GL_LIGHTING)

            if vertex_format == GL_C4F_N3F_V3F:
                glEnable(GL_COLOR_MATERIAL)
                glColorMaterial(face, GL_AMBIENT_AND_DIFFUSE)

            glInterleavedArrays(vertex_format, 0, material.gl_floats)
            glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

        glPopAttrib()
        glPopClientAttrib()

    def window_load(self, window):
        super(RenderObj, self).window_load(window)
        self.window = window
        self.result = 0
        self.render_cmd = Condition()
        self.render_lock = Lock()
        self.render_req = 0
        self.render_ack = Event()
        self.render_name = None if self.view_mode else 'render'
        if self.view_mode:
            self.render_ack.set()

    @contextmanager
    def matrix_trans(self, matrix: np.ndarray):
        glPushMatrix()
        glMultMatrixd(matrix.ctypes.data_as(POINTER(GLdouble)))
        yield
        glPopMatrix()

    @contextmanager
    def set_render_name(self, render_name, wait=False):
        with self.render_cmd:
            if self.closing:
                raise RuntimeError('window closing')
            self.render_req += 1
            self.post_event()
            self.render_cmd.wait()
            try:
                yield self
                self.render_name = render_name
            finally:
                self.render_lock.release()
        if wait:
            self.render_ack.wait()

    def set_fast_switching(self):
        self.result = 1 if self.imageid < len(conf.dblist) - 1 else 0
        glfw.set_window_should_close(self.window, GL_TRUE)
        self.post_event()
        return self.result

    def window_closing(self, window):
        super(RenderObj, self).window_closing(window)
        with self.render_cmd:
            if self.render_req > 0:
                self.render_cmd.notify_all()
        with self.render_lock:
            self.free_texture()
            if self.fast_switching():
                glfw.set_window_should_close(window, GL_FALSE)
                glfw.poll_events()
                self.update_scene(self.load_image(conf.data_dir))
                self.window_load(window)

    def free_texture(self):
        for mtl in chain.from_iterable(mesh.materials for mesh in self.scene.mesh_list):
            if mtl.texture:
                del mtl.texture.image

    def fast_switching(self):
        is_fast_switching = True
        if self.result == 1:
            print('Switching Ahead...', flush=True)
            self.imageid = min(self.imageid + 1, len(conf.dblist) - 1)
        elif self.result == 2:
            print('Switching Back...', flush=True)
            self.imageid = max(0, self.imageid - 1)
        else:
            print('Closing...', flush=True)
            is_fast_switching = False
        return is_fast_switching

    @contextmanager
    def render_locking(self):
        with super(RenderObj, self).render_locking():
            with self.render_cmd:
                if self.render_req > 0:
                    self.render_req -= 1
                    self.render_lock.acquire()
                    self.render_ack.clear()
                    self.render_cmd.notify()
            with self.render_lock:
                yield
                if self.render_name:
                    self.save_buffer(self.render_name)
                    self.render_ack.set()
                    self.render_name = None

    def save_buffer(self, im_name='render'):
        img = self.get_buffer()
        depth, stencil = self.get_depth_stencil()
        xyz, label = self.part_pointcloud(depth=depth, stencil=stencil)
        perm = np.random.permutation(label.size)
        im_id = conf.dblist[self.imageid]
        file_path = os.path.join(self.render_dir, im_id)
        with suppress(FileExistsError):
            os.mkdir(file_path)
        imwrite(os.path.join(file_path, f'{im_name}-RGB.png'), np.flipud(img))
        np.save(os.path.join(file_path, f'{im_name}-DEPTH.npy'), np.flipud(depth))
        np.save(os.path.join(file_path, f'{im_name}-XYZ.npy'), xyz[perm].astype(np.float32))
        np.save(os.path.join(file_path, f'{im_name}-LABEL.npy'),
                np.stack([label[perm].astype(np.int32), perm.astype(np.int32)], axis=0))
        imwrite(os.path.join(file_path, f'{im_name}-STENCIL.png'), np.flipud(stencil))
        m_trans = get_gl_matrix('PROJECTION'), get_gl_matrix('MODELVIEW')
        np.save(os.path.join(file_path, f'{im_name}-MATRIX.npy'), np.stack(m_trans, axis=0))


def main(idx, autogen=False):
    show = RenderObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
