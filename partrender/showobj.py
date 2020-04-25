import os
from contextlib import contextmanager
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront.visualization import draw_material, gl_light
from pywavefront.wavefront import Wavefront

from tools.cfgreader import conf


class ShowObj:
    post_event = staticmethod(glfw.post_empty_event)

    _BUFFER_TYPE: Final = {
        'GL_RGB': (GL_RGB, GLubyte, GL_UNSIGNED_BYTE, 3),
        'GL_DEPTH_COMPONENT': (GL_DEPTH_COMPONENT, GLfloat, GL_FLOAT, 1),
        'GL_STENCIL_INDEX': (GL_STENCIL_INDEX, GLubyte, GL_UNSIGNED_BYTE, 1)
    }

    def __init__(self, scene: Wavefront, title='ShowObj'):
        self._rot_angle_old = None
        self._cur_pos_old = (0.0, 0.0)
        self._cur_sel_idx = (GLubyte * 1)()
        self._selected_idx = None

        self.scene = scene
        self.title = title
        self.cur_rot_mode = False
        self.sel_set = set()
        self.del_set = set()
        self.viewport = (GLint * 4)()
        self.result = 0
        self.closing = False
        self.scale = None
        self.max_lights = 0

        self.rot_angle = None
        self.initial_look_at = None
        self.up_vector = None
        self.light_source = []

    @contextmanager
    def render_procedure(self):
        yield

    def default_param(self):
        self.rot_angle = np.array((38.0, -17.0), dtype=np.float32)
        self.initial_look_at = np.array((0, 0, 3), dtype=np.float32)
        self.up_vector = np.array((0, 1, 0), dtype=np.float32)
        self.clear_light_source()
        self.add_light_source()

    def add_light_source(self, *, ambient=(0.2, 0.2, 0.2, 1.0), diffuse=(1.0, 1.0, 1.0, 1.0),
                         specular=(1.0, 1.0, 1.0, 1.0), position=(0.0, 4.0, 3.0, 0.0)):
        self.light_source.append({
            GL_AMBIENT: gl_light(ambient),
            GL_DIFFUSE: gl_light(diffuse),
            GL_SPECULAR: gl_light(specular),
            GL_POSITION: gl_light(position)
        })

    def clear_light_source(self):
        self.light_source.clear()

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glGetIntegerv(GL_VIEWPORT, self.viewport)
        x, y, width, height = self.viewport
        gluPerspective(45.0, width / height, 1, 10.0)

    def viewpoint(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rot_x = np.radians(self.rot_angle[0])
        cosx, sinx = np.cos(rot_x), np.sin(rot_x)
        mrotx = np.array([[cosx, 0, sinx],
                          [0, 1, 0],
                          [-sinx, 0, cosx]])
        pv = mrotx @ self.initial_look_at
        px, py, pz = pv
        ux, uy, uz = self.up_vector
        gluLookAt(px, py, pz, 0.0, 0.0, 0.0, ux, uy, uz)

        x, y, z = np.cross(pv, self.up_vector)
        glRotatef(self.rot_angle[1], x, y, z)

    def lighting(self):
        glEnable(GL_LIGHTING)

        for i in range(self.max_lights):
            lid = GL_LIGHT0 + i
            if i < len(self.light_source):
                for k, v in self.light_source[i].items():
                    glLightfv(lid, k, v)
                glEnable(lid)
            else:
                glDisable(lid)

    def draw_model(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_STENCIL_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        cur_idx = self.get_cur_sel_idx()
        for idx, mesh in enumerate(self.scene.mesh_list):
            if idx in self.del_set:
                continue
            glStencilFunc(GL_ALWAYS, idx + 1, 0xFF)  # Set any stencil to 1
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
            glStencilMask(0xFF)  # Write to stencil buffer
            for material in mesh.materials:
                material.ambient = [0.6, 0.6, 0.6, 1.0]
                if idx == cur_idx and cur_idx != self._selected_idx:
                    material.ambient = [0.0, 1.0, 1.0, 1.0]
                elif idx in self.sel_set:
                    material.ambient = [0.0, 0.0, 1.0, 1.0]
                self.draw_material(idx, material)

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        draw_material(material, face, lighting_enabled, textures_enabled)

    def get_cur_sel_idx(self):
        [cur_idx] = self._cur_sel_idx
        return cur_idx - 1 if cur_idx else None

    def mouse_button_fun(self, window, button, action, mods):
        if button == 1:
            self._cur_pos_old = glfw.get_cursor_pos(window)
            self.cur_rot_mode = (action == 1)
            self._rot_angle_old = self.rot_angle
        elif button == 0 and action == 1:
            cur_idx = self.get_cur_sel_idx()
            self._selected_idx = cur_idx
            if cur_idx in self.sel_set:
                self.sel_set.remove(cur_idx)
            else:
                self.sel_set.add(cur_idx)

    def cursor_pos_fun(self, window, xpos, ypos):
        if self.cur_rot_mode:
            offset = np.array((xpos, ypos)) - self._cur_pos_old
            self.rot_angle = self._rot_angle_old - offset
        else:
            x, y, width, height = self.viewport
            glReadPixels(int(xpos * self.scale),
                         int(height - ypos * self.scale), 1, 1,
                         GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, self._cur_sel_idx)
            cur_idx = self.get_cur_sel_idx()
            if self._selected_idx != cur_idx:
                self._selected_idx = None

    def scroll_fun(self, window, xoffset, yoffset):
        self.rot_angle += np.array((xoffset, yoffset))

    def key_fun(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_D:
                self.del_set.update(self.sel_set)
            elif key == glfw.KEY_N:
                self.result = 1
                glfw.set_window_should_close(window, GL_TRUE)
            elif key == glfw.KEY_P:
                self.result = 2
                glfw.set_window_should_close(window, GL_TRUE)
            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(window, GL_TRUE)
            elif key == glfw.KEY_SPACE:
                cur_idx = self.get_cur_sel_idx()
                if not cur_idx:
                    return
                self.do_part(cur_idx)
            elif key == glfw.KEY_S:
                img = self.get_buffer()
                imwrite('render.png', img)

    def window_size_fun(self, window, width, height):
        glViewport(0, 0, int(width * self.scale), int(height * self.scale))

    def window_load(self, window):
        glfw.set_window_title(window, self.title)
        self.default_param()
        self.closing = False

    def window_closing(self, window):
        glfw.set_window_title(window, 'Closing... ' + self.title)
        self.closing = True

    def show_obj(self):
        # Initialize the library
        if not glfw.init():
            raise RuntimeError('An error occurred')

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 800, self.title, None, None)
        if not window:
            raise RuntimeError('An error occurred')

        # Make the window's context current
        glfw.make_context_current(window)

        glfw.set_mouse_button_callback(window, self.mouse_button_fun)
        glfw.set_cursor_pos_callback(window, self.cursor_pos_fun)
        glfw.set_scroll_callback(window, self.scroll_fun)
        glfw.set_key_callback(window, self.key_fun)
        glfw.set_window_size_callback(window, self.window_size_fun)

        max_lights = (GLint * 1)()
        glGetIntegerv(GL_MAX_LIGHTS, max_lights)
        [self.max_lights] = max_lights
        glGetIntegerv(GL_VIEWPORT, self.viewport)

        fw, fh = glfw.get_framebuffer_size(window)
        ww, wh = glfw.get_window_size(window)
        self.scale = fw / ww
        assert fh / wh == self.scale

        self.window_load(window)
        # Loop until the user closes the window
        while not glfw.window_should_close(window) or self.result > 0:
            with self.render_procedure():
                self.perspective()
                self.viewpoint()

                self.lighting()
                # Render here, e.g. using pyOpenGL
                self.draw_model()

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.wait_events()

            if glfw.window_should_close(window):
                self.window_closing(window)

        assert self.result <= 0, 'NOT POSSIBLE'
        glfw.terminate()

    def do_part(self, partid):
        pass

    def get_buffer(self, buf_type_str='GL_RGB'):
        x, y, width, height = self.viewport
        buf_type, buf_ctypes, buf_data_type, ch = self._BUFFER_TYPE.get(buf_type_str)
        buf_shape = (height, width) if ch == 1 else (height, width, ch)
        buf = (buf_ctypes * (width * height * ch))()
        glReadPixels(x, y, width, height, buf_type, buf_data_type, buf)
        buf = np.ctypeslib.as_array(buf).reshape(buf_shape)
        return np.flip(buf, axis=0)


def main(idx):
    while True:
        im_id = conf.dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ShowObj(scene)
        show.show_obj()
        if show.result == 1:
            idx = min(idx + 1, len(conf.dblist) - 1)
        elif show.result == 2:
            idx = max(0, idx - 1)
        else:
            break


if __name__ == '__main__':
    main(0)
