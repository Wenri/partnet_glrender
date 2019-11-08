import glfw
from pyglet.gl import *
from pywavefront import visualization
import numpy as np

class ShowObj:
    def __init__(self, scene):
        self.scene = scene
        self.rot_x = 0;
        self.rot_y = 0;

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        viewport = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, viewport)
        x, y, width, height = viewport
        gluPerspective(45.0, width / height, 1, 10.0)

    def viewpoint(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

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

        # glDepthFunc(GL_LESS)
        # glEnable(GL_DEPTH_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        visualization.draw(self.scene, lighting_enabled=True)

        # glutSolidTeapot(1)

    def mouse_button_fun(self, window, button, action, mods):
        print(button)

    def cursor_pos_fun(self, window, xpos, ypos):
        pass

    def scroll_fun(self, window, xoffset, yoffset):
        glRotatef(xoffset, 0.0, 1.0, 0.0)
        glRotatef(yoffset, 1.0, 0.0, 0.0)
        m = (GLfloat * 16)()
        glGetFloatv(GL_MODELVIEW_MATRIX, m)
        m = np.ctypeslib.as_array(m).reshape((4, 4))
        print()

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
        self.viewpoint()

        # Loop until the user closes the window
        while not glfw.window_should_close(window):
            # Render here, e.g. using pyOpenGL
            self.draw_model()

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.wait_events()

        glfw.terminate()
