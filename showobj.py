import glfw
from pyglet.gl import *
from pywavefront import visualization


class ShowObj:
    def __init__(self, scene):
        # Initialize the library
        if not glfw.init():
            raise Exception('An error occurred')

        self.scene = scene

    def __del__(self):
        glfw.terminate()

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
        gluLookAt(0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

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

    def show_obj(self):
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(640, 480, "Hello World", None, None)
        if not window:
            raise Exception('An error occurred')

        # Make the window's context current
        glfw.make_context_current(window)

        # Loop until the user closes the window
        while not glfw.window_should_close(window):
            # Render here, e.g. using pyOpenGL
            self.perspective()
            self.viewpoint()
            self.draw_model()

            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.wait_events()
