import os
from contextlib import contextmanager
from threading import Lock, Condition, Event
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light
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

        super().__init__(self.load_image())

    def load_image(self):
        im_id = conf.dblist[self.imageid]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        for material in scene.materials.values():
            material.ambient = [0.2, 0.2, 0.2, 1.0]
        self.lock_list = [Lock() for i in range(len(scene.mesh_list))]
        return scene

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        """Draw a single material"""

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHT0)
        glCullFace(GL_BACK)

        with self.lock_list[idx]:
            if material.gl_floats is None:
                material.gl_floats = np.asarray(material.vertices, dtype=np.float32).ctypes
                material.triangle_count = len(material.vertices) / material.vertex_size

            vertex_format = self._VERTEX_FORMATS.get(material.vertex_format)
            if not vertex_format:
                raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

            glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
            glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
            glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
            glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
            glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))

            if material.has_normals:
                glEnable(GL_LIGHTING)
            else:
                glDisable(GL_LIGHTING)

            if vertex_format == GL_C4F_N3F_V3F and self.view_mode:
                glEnable(GL_COLOR_MATERIAL)

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
        self.render_name = None
        self.look_at_reset()
        if not self.view_mode:
            self.render_name = 'render'
            self.del_set = set()
            self.sel_set = set()

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
            if self.fast_switching():
                self.scene = self.load_image()
                glfw.set_window_should_close(window, GL_FALSE)
                self.window_load(window)

    def fast_switching(self):
        is_fast_switching = True
        if self.result == 1:
            self.imageid = min(self.imageid + 1, len(conf.dblist) - 1)
        elif self.result == 2:
            self.imageid = max(0, self.imageid - 1)
        else:
            is_fast_switching = False
        return is_fast_switching

    def show_obj(self):
        super().show_obj()

    def draw_model(self):
        with self.render_cmd:
            if self.render_req > 0:
                self.render_req -= 1
                self.render_lock.acquire()
                self.render_ack.clear()
                self.render_cmd.notify()
        with self.render_lock:
            super().draw_model()
            if self.render_name:
                self.save_buffer(self.render_name)
                self.render_ack.set()
                self.render_name = None

    def save_buffer(self, im_name):
        img, depth, stencil = map(self.get_buffer, ('GL_RGB', 'GL_DEPTH_COMPONENT', 'GL_STENCIL_INDEX'))
        im_id = conf.dblist[self.imageid]
        file_path = os.path.join(self.render_dir, im_id)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        imwrite(os.path.join(file_path, f'{im_name}-RGB.png'), img)
        savemat(os.path.join(file_path, f'{im_name}-DEPTH.mat'), {'depth': depth}, do_compression=True)
        imwrite(os.path.join(file_path, f'{im_name}-STENCIL.png'), stencil)
