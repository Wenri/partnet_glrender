from contextlib import contextmanager
from functools import partial
from typing import Final

import glfw
import numpy as np
from imageio import imwrite
from pyglet.gl import *
from pywavefront.visualization import draw_material, gl_light
from pywavefront.wavefront import Wavefront


def get_gl_matrix(matrix_type_str='MODELVIEW'):
    import pyglet.gl
    buf = (GLdouble * 16)()
    matrix_type = getattr(pyglet.gl, f'GL_{matrix_type_str}_MATRIX')
    glGetDoublev(matrix_type, buf)
    buf = np.ctypeslib.as_array(buf)
    buf.shape = (4, 4)
    return buf


class ShowObj(object):
    post_event = staticmethod(glfw.post_empty_event)

    _BUFFER_TYPE: Final = {
        'GL_RGB': (GL_RGB, GLubyte, GL_UNSIGNED_BYTE, 3),
        'GL_DEPTH_COMPONENT': (GL_DEPTH_COMPONENT, GLfloat, GL_FLOAT, None),
        'GL_STENCIL_INDEX': (GL_STENCIL_INDEX, GLubyte, GL_UNSIGNED_BYTE, None),
        'GL_DEPTH24_STENCIL8': (GL_DEPTH_STENCIL, GLuint, GL_UNSIGNED_INT_24_8, None),
        'GL_DEPTH32F_STENCIL8': (GL_DEPTH_STENCIL, GLuint, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 2)
    }

    def __init__(self, scene: Wavefront, title='ShowObj'):
        self._rot_angle_old = None
        self._cur_pos_old = (0.0, 0.0)
        self._cur_sel_idx = (GLubyte * 1)()
        self._selected_idx = None

        self.scene = None
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
        self.key_press_dispatcher = {}
        self.default_key_func()
        self.update_scene(scene)

    def update_scene(self, scene):
        old_scene, self.scene = self.scene, scene
        return old_scene

    @contextmanager
    def render_locking(self):
        yield

    def default_param(self):
        self.rot_angle = np.array((38.0, -17.0), dtype=np.float32)
        self.initial_look_at = np.array((0, 0, 3), dtype=np.float32)
        self.up_vector = np.array((0, 1, 0), dtype=np.float32)
        self.clear_light_source()
        self.add_light_source()
        self.del_set.clear()
        self.sel_set.clear()

    def default_key_func(self):
        self.key_press_dispatcher.clear()
        self.act_key('D', lambda w: self.del_set.update(self.sel_set))
        self.act_key('N', partial(self.close_with_result, result=1))
        self.act_key('P', partial(self.close_with_result, result=2))
        self.act_key('Q', partial(glfw.set_window_should_close, value=GL_TRUE))
        self.act_key('S', lambda w: imwrite('render.png', np.flipud(self.get_buffer())))
        self.act_key('SPACE', lambda w: self.do_part(cur_idx) if (cur_idx := self.cur_sel()) is not None else print(
            'No selection'))

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
        gluPerspective(45.0, width / height, 1.0, 5.0)

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
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        for i in range(self.max_lights):
            lid = GL_LIGHT0 + i
            if i < len(self.light_source):
                for k, v in self.light_source[i].items():
                    glLightfv(lid, k, v)
                glEnable(lid)
            else:
                glDisable(lid)

    def draw_model(self):
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_STENCIL_TEST)
        glEnable(GL_MULTISAMPLE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        cur_idx = self.cur_sel()
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

    def cur_sel(self):
        [cur_idx] = self._cur_sel_idx
        return cur_idx - 1 if cur_idx else None

    def mouse_button_fun(self, window, button, action, mods):
        if button == 1:
            self._cur_pos_old = glfw.get_cursor_pos(window)
            self.cur_rot_mode = (action == 1)
            self._rot_angle_old = self.rot_angle
        elif button == 0 and action == 1:
            cur_idx = self.cur_sel()
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
            cur_idx = self.cur_sel()
            if self._selected_idx != cur_idx:
                self._selected_idx = None

    def scroll_fun(self, window, xoffset, yoffset):
        self.rot_angle += np.array((xoffset, yoffset))

    def key_fun(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if func := self.key_press_dispatcher.get(key):
                func(window)

    def act_key(self, key_str, func):
        key = getattr(glfw, f'KEY_{key_str}')
        if key in self.key_press_dispatcher:
            raise RuntimeError(f'Duplicate key {key_str}')
        self.key_press_dispatcher[key] = func

    def register_key_func(self, key_str):
        def _wrapper(func):
            self.act_key(key_str=key_str, func=func)
            return func

        return _wrapper

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
        with self._setup_window() as window:
            self._update_gl_variable(window)
            self.window_load(window)
            self._main_loop(window)

        return self.result

    def close_with_result(self, window, result=0):
        self.result = result
        glfw.set_window_should_close(window, GL_TRUE)

    @staticmethod
    def _hint_gl_version(version=(2, 1)):  # macOS supported version: 2.1, 3.2, 4.1
        glfw.window_hint(0x0002100D, 16)  # GLFW_SAMPLES
        glfw.window_hint(0x00022002, version[0])  # GLFW_CONTEXT_VERSION_MAJOR
        glfw.window_hint(0x00022003, version[1])  # GLFW_CONTEXT_VERSION_MINOR
        if version[0] > 2:
            glfw.window_hint(0x00022006, GL_TRUE)  # GLFW_OPENGL_FORWARD_COMPAT
            glfw.window_hint(0x00022008, 0x00032001)  # GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE

    @contextmanager
    def _setup_window(self):
        # Initialize the library
        glfw.ERROR_REPORTING = 'warn'
        if not glfw.init():
            raise RuntimeError('An error occurred when calling glfw.init')

        for monitor in glfw.get_monitors():
            print(f"'{glfw.get_monitor_name(monitor).decode()}' {glfw.get_monitor_content_scale(monitor)}", end='. ')
        print(f"GLFW Ver {glfw.get_version_string().decode()}.")

        self._hint_gl_version()

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 800, self.title, None, None)
        if not window:
            raise RuntimeError('An error occurred when creating window')

        # Make the window's context current
        glfw.make_context_current(window)

        glfw.set_mouse_button_callback(window, self.mouse_button_fun)
        glfw.set_cursor_pos_callback(window, self.cursor_pos_fun)
        glfw.set_scroll_callback(window, self.scroll_fun)
        glfw.set_key_callback(window, self.key_fun)
        glfw.set_window_size_callback(window, self.window_size_fun)
        glfw.set_window_refresh_callback(window, self._update_gl_variable)

        yield window

        glfw.terminate()

    def _update_gl_variable(self, window):
        # max_lights
        max_lights = (GLint * 1)()
        glGetIntegerv(GL_MAX_LIGHTS, max_lights)
        [self.max_lights] = max_lights

        # viewport
        glGetIntegerv(GL_VIEWPORT, self.viewport)

        # scale
        fw, fh = glfw.get_framebuffer_size(window)
        ww, wh = glfw.get_window_size(window)
        self.scale = fw / ww
        assert fh / wh == self.scale, "Non square scale"

    def _main_loop(self, window):
        # Loop until the user closes the window
        while not glfw.window_should_close(window) or self.result > 0:
            with self.render_locking():
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

    def do_part(self, part_id):
        """
        Action on selected part
        :type part_id: int
        :param part_id:
        :return: None
        """
        print(f'Showing {part_id=}')
        hmc = self.part_pointcloud(part_id)
        print(np.max(hmc, axis=1), np.min(hmc, axis=1))

    def part_pointcloud(self, part_id=-1, depth=None, stencil=None):
        """
        only support Perspective Projection
        :type part_id: int
        :param part_id: part id for stencil test
        :param depth: optional depth buffer
        :param stencil: optional stencil buffer
        :return: xyz coordinates in world for rendered points
        """
        if depth is None:
            depth, stencil = self.get_depth_stencil()

        if part_id < 0:
            pos = np.nonzero(stencil)
        else:
            pos = np.nonzero(stencil == part_id + 1)

        depth = depth[pos].astype(np.float64)
        label = stencil[pos]

        m_trans = get_gl_matrix('PROJECTION')
        assert np.allclose(m_trans[:2, 2:], 0) and np.allclose(m_trans[2:, -1], [-1.0, 0.0])

        m_a, m_b = m_trans[2:, 2]  # A, B in Projection Matrix
        depth *= 2.0
        depth -= 1.0  # Inverse viewport transform
        neg_z = m_b / (depth + m_a)  # Reverse projection transform

        pos = np.array(pos, dtype=np.float64)
        pos += 0.5  # Half pixel tricks
        pos *= 2.0 / np.array(stencil.shape)[:, np.newaxis]
        pos -= 1.0  # Inverse viewport transform
        pos *= neg_z  # Normalized Device Coordinates (NDC) to Clip Coordinates
        depth *= neg_z  # Normalized Device Coordinates (NDC) to Clip Coordinates

        xyz = np.concatenate((np.flipud(pos), depth[np.newaxis, :], neg_z[np.newaxis, :]), axis=0)
        m_trans = np.matmul(get_gl_matrix('MODELVIEW'), m_trans)  # get combined transform matrix
        xyz = np.linalg.solve(m_trans.T.astype(np.float64), xyz)  # Inverse transform
        return xyz.T, label

    def point_is_visible(self, p):
        n, d = p.shape
        assert d == 3
        p = np.concatenate([p, np.ones(shape=[n, 1])], axis=-1)
        m_trans = get_gl_matrix('PROJECTION')
        m_trans = np.matmul(get_gl_matrix('MODELVIEW'), m_trans)  # get combined transform matrix
        p = np.matmul(p, m_trans.astype(np.float64))

        depth, stencil = self.get_depth_stencil()

        xyz = p[:, :3] / p[:, 3][:, np.newaxis]
        pos = xyz[:, :2]
        in_screen_mask = np.logical_and(np.all(pos >= -1.0, axis=-1), np.all(pos < 1.0, axis=-1))
        pos = np.fliplr(pos[in_screen_mask, :])
        pos += 1.0
        pos /= 2.0
        pos *= np.array(stencil.shape)
        w, h = pos.T.astype(np.int)

        d = xyz[in_screen_mask, 2]
        d += 1.0
        d /= 2.0

        ret = np.ones(shape=(n,), dtype=np.bool)
        ret[in_screen_mask] = np.logical_or(stencil[w, h] == 0, d < depth[w, h])
        return ret

    def get_buffer(self, buf_type_str='GL_RGB'):
        x, y, width, height = self.viewport
        buf_type, buf_ctypes, buf_data_type, ch = self._BUFFER_TYPE.get(buf_type_str)
        buf = (buf_ctypes * (width * height * ch))()
        glReadPixels(x, y, width, height, buf_type, buf_data_type, buf)
        buf = np.ctypeslib.as_array(buf)
        buf.shape = (height, width, ch) if ch else (height, width)
        return buf

    def get_depth_stencil(self):
        import numpy.lib.stride_tricks as tricks
        stencil = self.get_buffer('GL_DEPTH32F_STENCIL8')
        depth = np.frombuffer(stencil.data, np.float32).reshape(stencil.shape)[:, :, 0]
        stencil = tricks.as_strided(np.frombuffer(stencil.data, np.uint8, offset=4), depth.shape, depth.strides)
        return depth, stencil


def main(im_file):
    scene = Wavefront(im_file)
    show = ShowObj(scene)
    show.show_obj()
    print(f'{show.result=}')


if __name__ == '__main__':
    main('/mybookduo/research/box.obj')
