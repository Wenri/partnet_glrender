import os
from threading import Lock, Condition
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront import Wavefront
from pywavefront.visualization import gl_light

from partrender.showobj import ShowObj
from tools.cfgreader import conf


class RenderObj(ShowObj):
    VERTEX_FORMATS: Final = {
        'V3F': GL_V3F,
        'C3F_V3F': GL_C3F_V3F,
        'N3F_V3F': GL_N3F_V3F,
        'T2F_V3F': GL_T2F_V3F,
        'C4F_N3F_V3F': GL_C4F_N3F_V3F,
        'T2F_C3F_V3F': GL_T2F_C3F_V3F,
        'T2F_N3F_V3F': GL_T2F_N3F_V3F,
        'T2F_C4F_N3F_V3F': GL_T2F_C4F_N3F_V3F,
    }

    def __init__(self, start_id, auto_generate=False):
        self.imageid = start_id
        self.cluster_color = not auto_generate
        self.render_lock = None
        self.render_cmd = None
        self.render_req = 0
        self.render_ack = None
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

            vertex_format = self.VERTEX_FORMATS.get(material.vertex_format)
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

            if vertex_format == GL_C4F_N3F_V3F and self.cluster_color:
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
        self.render_ack = None
        self.look_at_reset()
        if not self.cluster_color:
            self.render_ack = 'render'
            self.del_set = set()
            self.sel_set = set()

    def set_render_name(self, render_name):
        return RenderNotify(self, render_name)

    def window_closing(self, window):
        super(RenderObj, self).window_closing(window)
        with self.render_cmd:
            self.render_cmd.notify_all()
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
                self.render_cmd.notify()
        with self.render_lock:
            super().draw_model()
            if self.render_ack:
                img = self.save_to_buffer()
                im_id = conf.dblist[self.imageid]
                im_name = '{}.png'.format(self.render_ack)
                file_path = os.path.join(conf.render_dir, im_id)
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                imwrite(os.path.join(file_path, im_name), img)
                self.render_ack = None


class RenderNotify:
    def __init__(self, target: RenderObj, render_name):
        self.render_name = render_name
        self.target = target

    def __enter__(self):
        self.target.render_cmd.acquire()
        self.target.render_req += 1
        glfw.post_empty_event()
        self.target.render_cmd.wait()
        if self.target.closing:
            raise RuntimeError('window closing')
        self.target.render_ack = self.render_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.target.render_lock.release()
        self.target.render_cmd.release()
