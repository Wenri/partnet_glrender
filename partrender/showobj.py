import os

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront.visualization import draw_material
from pywavefront.wavefront import Wavefront

from tools.cfgreader import conf


class ShowObj:
    def __init__(self, scene: Wavefront, title='ShowObj'):
        self.scene = scene
        self.title = title
        self.cur_pos_old = (0.0, 0.0)
        self.cur_rot_mode = False
        self.cur_sel_idx = (GLubyte * 1)(0xFF)
        self.invalid_cur_idx = 0xFF
        self.sel_set = set()
        self.del_set = set()
        self.viewport = (GLint * 4)()
        self.result = 0
        self.closing = False
        self.scale = None

        self.rot_angle = None
        self.initial_look_at = None
        self.up_vector = None
        self.look_at_reset()

        self.rot_angle_old = self.rot_angle

    def look_at_reset(self):
        self.rot_angle = np.array((38.0, -17.0), dtype=np.float32)
        self.initial_look_at = np.array((0, 0, 3), dtype=np.float32)
        self.up_vector = np.array((0, 1, 0), dtype=np.float32)

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

    def material(self):
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = [50.0]
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

    def lighting(self):

        light_ambient = (GLfloat * 4)(0.2, 0.2, 0.2, 1.0)
        light_diffuse = (GLfloat * 4)(1.0, 1.0, 1.0, 1.0)
        light_specular = (GLfloat * 4)(1.0, 1.0, 1.0, 1.0)
        light_position = (GLfloat * 4)(0.0, 4.0, 3.0, 0.0)

        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def draw_model(self):
        self.lighting()
        # self.material()

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_STENCIL_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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
                if idx == cur_idx and cur_idx != self.invalid_cur_idx:
                    material.ambient = [0.0, 1.0, 1.0, 1.0]
                elif idx in self.sel_set:
                    material.ambient = [0.0, 0.0, 1.0, 1.0]
                self.draw_material(idx, material)

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        draw_material(material, face, lighting_enabled, textures_enabled)

    def get_cur_sel_idx(self):
        cur_idx, = self.cur_sel_idx
        if cur_idx == 0 or cur_idx == 0xFF:
            return 0xFF
        return cur_idx - 1

    def mouse_button_fun(self, window, button, action, mods):
        if button == 1:
            self.cur_pos_old = glfw.get_cursor_pos(window)
            self.cur_rot_mode = (action == 1)
            self.rot_angle_old = self.rot_angle
        elif button == 0 and action == 1:
            cur_idx = self.get_cur_sel_idx()
            self.invalid_cur_idx = cur_idx
            if cur_idx in self.sel_set:
                self.sel_set.remove(cur_idx)
            else:
                self.sel_set.add(cur_idx)

    def cursor_pos_fun(self, window, xpos, ypos):
        if self.cur_rot_mode:
            offset = np.array((xpos, ypos)) - self.cur_pos_old
            self.rot_angle = self.rot_angle_old - offset
        else:
            x, y, width, height = self.viewport
            glReadPixels(int(xpos * self.scale),
                         int(height - ypos * self.scale), 1, 1,
                         GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, self.cur_sel_idx)
            cur_idx = self.get_cur_sel_idx()
            if self.invalid_cur_idx != cur_idx:
                self.invalid_cur_idx = 0xFF

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
                if cur_idx == 0xff:
                    return
                self.do_part(cur_idx)
            elif key == glfw.KEY_S:
                img = self.save_to_buffer()
                imwrite('render.png', img)

    def window_size_fun(self, window, width, height):
        glViewport(0, 0, int(width * self.scale), int(height * self.scale))

    def window_load(self, window):
        glfw.set_window_title(window, self.title)
        self.closing = False

    def window_closing(self, window):
        glfw.set_window_title(window, 'Closing... ' + self.title)
        self.closing = True

    def show_obj(self):
        # Initialize the library
        if not glfw.init():
            raise Exception('An error occurred')

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 800, self.title, None, None)
        if not window:
            raise Exception('An error occurred')

        # Make the window's context current
        glfw.make_context_current(window)

        glfw.set_mouse_button_callback(window, self.mouse_button_fun)
        glfw.set_cursor_pos_callback(window, self.cursor_pos_fun)
        glfw.set_scroll_callback(window, self.scroll_fun)
        glfw.set_key_callback(window, self.key_fun)
        glfw.set_window_size_callback(window, self.window_size_fun)

        glGetIntegerv(GL_VIEWPORT, self.viewport)
        fw, fh = glfw.get_framebuffer_size(window)
        ww, wh = glfw.get_window_size(window)
        self.scale = fw / ww
        assert fh / wh == self.scale

        self.window_load(window)
        # Loop until the user closes the window
        while not glfw.window_should_close(window):
            self.perspective()
            self.viewpoint()

            # Render here, e.g. using pyOpenGL
            self.draw_model()

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.wait_events()

            if glfw.window_should_close(window):
                self.window_closing(window)

        glfw.terminate()

    def do_part(self, partid):
        pass

    def save_to_buffer(self) -> np.ndarray:
        x, y, width, height = self.viewport
        buf = (GLubyte * (width * height * 3))(0)
        glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, buf)
        buf = np.ctypeslib.as_array(buf).reshape((height, width, 3))
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