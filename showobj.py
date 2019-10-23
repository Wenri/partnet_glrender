import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from geometry_utils import *


class ShowObj:
    def __init__(self, obj_parts):
        # Initialize the library
        if not glfw.init():
            raise Exception('An error occurred')

        self.obj_buffers = [VertexBuffer(n, v, f-1) for n, v, f
                            in obj_parts]

    def __del__(self):
        del self.obj_buffers
        glfw.terminate()

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        x, y, width, height = glGetInteger(GL_VIEWPORT)
        gluPerspective(45.0, width/height, 1, 10.0)

    def viewpoint(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def material(self):
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = [50.0]
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

    def lighting(self):

        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        light_position = [1.0, 1.0, -4.0, 0.0]

        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    def draw_model(self):
        self.lighting()
        self.material()

        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        for b in self.obj_buffers:
            print(b.name)
            b.draw()
            break

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
            glfw.poll_events()


class VertexBuffer:
    def __init__(self, name, v, f):
        self.name = name

        self.fn = np.stack([tri_normal(v[v1], v[v2], v[v3])
                            for v1, v2, v3 in f])

        def vec_normal(vid):
            ind = np.any(f == vid, axis=1)
            if not np.any(ind):
                return np.array([0., 0., 0.], dtype=np.float32)
            norm_vec = np.mean(self.fn[ind, :], axis=0)
            norm = np.linalg.norm(norm_vec)
            return norm_vec if norm == 0 else norm_vec / norm

        self.vn = np.stack([vec_normal(i) for i in range(v.size)])
        self.v, self.f = v,f
        self.vbo = vbo.VBO(np.concatenate((v, self.vn)))
        self.ibo = vbo.VBO(f, 'GL_STATIC_DRAW',
                           'GL_ELEMENT_ARRAY_BUFFER')

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        with self.vbo:
            glVertexPointer(3, GL_FLOAT, GL_FALSE, None)
            glNormalPointer(GL_FLOAT, GL_FALSE,
                            c_void_p(self.v.nbytes))
        with self.ibo:
            glDrawElements(GL_TRIANGLES, self.ibo.data.size,
                           GL_UNSIGNED_INT, None)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
