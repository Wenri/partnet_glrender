import logging
import os
import sys
from contextlib import contextmanager, ExitStack
from ctypes import POINTER
from functools import partial
from itertools import chain
from threading import Lock, Condition, Event
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light, bind_texture
from scipy.io import savemat

from partrender.showobj import ShowObj
from tools.cfgreader import conf


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
        scene = Wavefront(im_file)

        for material in scene.materials.values():
            material.ambient = [0.2, 0.2, 0.2, 1.0]
        return scene

    def update_scene(self, scene):
        old_scene = super(RenderObj, self).update_scene(scene)
        self.lock_list = [Lock() for _ in range(len(scene.mesh_list))]
        return old_scene

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
        im_id = conf.dblist[self.imageid]
        file_path = os.path.join(self.render_dir, im_id)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        imwrite(os.path.join(file_path, f'{im_name}-RGB.png'), np.flipud(img))
        savemat(os.path.join(file_path, f'{im_name}-DEPTH.mat'),
                {'depth': np.flipud(depth), 'xyz': xyz, 'label': label}, do_compression=True)
        imwrite(os.path.join(file_path, f'{im_name}-STENCIL.png'), np.flipud(stencil))


def main(idx, autogen=False):
    show = RenderObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
