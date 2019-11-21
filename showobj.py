import glfw
from pyglet.gl import *
from pywavefront.visualization import draw_material
import numpy as np


class ShowObj:
    def __init__(self, scene):
        self.scene = scene
        self.rot_angle = np.array((0.0, 0.0))
        self.rot_angle_old = self.rot_angle
        self.cur_pos_old = (0.0, 0.0)
        self.cur_rot_mode = False
        self.cur_name = 255
        self.viewport = (GLint * 4)()

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glGetIntegerv(GL_VIEWPORT, self.viewport)
        x, y, width, height = self.viewport
        gluPerspective(45.0, width / height, 1, 10.0)

    def viewpoint(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        rx = np.radians(self.rot_angle[0])
        cosx, sinx = np.cos(rx), np.sin(rx)
        mrotx = np.array([[cosx, 0, sinx],
                          [0, 1, 0],
                          [-sinx, 0, cosx]])
        px, py, pz = mrotx @ np.array([0, 2, 4])
        gluLookAt(px, py, pz, 0.0, 0.0, 0.0, 0, 1, 0)

        m = (GLfloat * 16)()
        glGetFloatv(GL_MODELVIEW_MATRIX, m)
        m = np.ctypeslib.as_array(m).reshape((4, 4))
        x, y, z, _ = m[0]
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
        light_position = (GLfloat * 4)(0.0, 4.0, 0.0, 0.0)

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

        for idx, mesh in enumerate(self.scene.mesh_list):
            glStencilFunc(GL_ALWAYS, idx, 0xFF)  # Set any stencil to 1
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
            glStencilMask(0xFF)  # Write to stencil buffer
            for material in mesh.materials:
                if idx == self.cur_name:
                    material.ambient = [0.0, 1.0, 1.0, 1.0]
                else:
                    material.ambient = [1.0, 1.0, 1.0, 1.0]
                draw_material(material)

        # glutSolidTeapot(1)

    def mouse_button_fun(self, window, button, action, mods):
        if button == 1:
            self.cur_pos_old = glfw.get_cursor_pos(window)
            self.cur_rot_mode = (action == 1)
            self.rot_angle_old = self.rot_angle

    def cursor_pos_fun(self, window, xpos, ypos):
        if self.cur_rot_mode:
            offset = np.array((xpos, ypos)) - self.cur_pos_old
            offset[0] = -offset[0]
            self.rot_angle = self.rot_angle_old + offset
        else:
            x, y, width, height = self.viewport
            cbuf = (GLubyte * 1)()
            glReadPixels(int(xpos), int(height - ypos), 1, 1,
                         GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, cbuf)
            self.cur_name, = cbuf
            print(xpos, ypos, self.cur_name)

    def scroll_fun(self, window, xoffset, yoffset):
        self.rot_angle += np.array((xoffset, yoffset))

    def show_obj(self):
        # Initialize the library
        if not glfw.init():
            raise Exception('An error occurred')

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(640, 480, "Hello World", None, None)
        if not window:
            raise Exception('An error occurred')

        # Make the window's context current
        glfw.make_context_current(window)

        glfw.set_mouse_button_callback(window, self.mouse_button_fun)
        glfw.set_cursor_pos_callback(window, self.cursor_pos_fun)
        glfw.set_scroll_callback(window, self.scroll_fun)

        self.perspective()

        # Loop until the user closes the window
        while not glfw.window_should_close(window):
            self.viewpoint()

            # Render here, e.g. using pyOpenGL
            self.draw_model()

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.wait_events()

        glfw.terminate()
