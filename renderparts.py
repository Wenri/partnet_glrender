import os
import time

from pywavefront import Wavefront
from pywavefront.mesh import Mesh
from pywavefront.material import Material
from cfgreader import conf
import numpy as np
from numpy.linalg import norm
from showobj import ShowObj
from threading import Thread, Lock
from pyglet.gl import *
from pywavefront.visualization import gl_light
from clusterpartnet import BkThread
import glfw


class ClsObj(ShowObj):
    VERTEX_FORMATS = {
        'V3F': GL_V3F,
        'C3F_V3F': GL_C3F_V3F,
        'N3F_V3F': GL_N3F_V3F,
        'T2F_V3F': GL_T2F_V3F,
        'C4F_N3F_V3F': GL_C4F_N3F_V3F,
        'T2F_C3F_V3F': GL_T2F_C3F_V3F,
        'T2F_N3F_V3F': GL_T2F_N3F_V3F,
        'T2F_C4F_N3F_V3F': GL_T2F_C4F_N3F_V3F,
    }

    def __init__(self, scene: Wavefront, grp_set: set):
        self.bkt = BkThread(scene.mesh_list, grp_set, glfw.post_empty_event)
        super().__init__(scene)

    def do_part(self, partid):
        mesh = self.scene.mesh_list[partid]
        vv = self.bkt.result_list[partid]
        # vv = np.array([1, 1, 1], dtype=np.float32)
        vv /= norm(vv)
        vx, vy, vz = vv
        rx, ry, rz = self.initial_look_at / norm(self.initial_look_at)
        print('\t'.join(
            "{}({}): ".format(mtl.name, mtl.vertex_format) for mtl in mesh.materials
        ), "{} {} {} ".format(vx, vy, vz), end='')
        if vx ** 2 + vz ** 2 > 0.01:
            cos_x = vx * rx + vz * rz
            rot_x = np.arccos(cos_x)
            _, dir_x, _ = np.cross([vx, 0, vz], [rx, 0, rz])
            if dir_x > 0:
                rot_x = 2 * np.pi - rot_x
        else:
            rot_x = 0
        cosx, sinx = np.cos(rot_x), np.sin(rot_x)
        mrotx = np.array([[cosx, 0, sinx],
                          [0, 1, 0],
                          [-sinx, 0, cosx]])
        rv = mrotx @ np.array([rx, ry, rz])
        rot_y = np.arccos(np.dot(rv, vv))
        x_x, _, x_z = np.cross(rv, vv)
        _, dir_y, _ = np.cross([x_x, 0, x_z], [rx, 0, rz])
        if dir_y > 0:
            rot_y = - rot_y
        self.rot_angle[0] = np.rad2deg(rot_x)
        self.rot_angle[1] = np.rad2deg(rot_y)
        print(self.rot_angle)

    def draw_material(self, idx, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        """Draw a single material"""
        self.bkt.lock_list[idx].acquire(blocking=True)

        if material.gl_floats is None:
            material.gl_floats = np.asarray(material.vertices, dtype=np.float32).ctypes
            material.triangle_count = len(material.vertices) / material.vertex_size

        vertex_format = self.VERTEX_FORMATS.get(material.vertex_format)
        if not vertex_format:
            raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)

        glDisable(GL_TEXTURE_2D)
        glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
        glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
        glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
        glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
        glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))
        glEnable(GL_LIGHT0)

        if material.has_normals:
            glEnable(GL_LIGHTING)
        else:
            glDisable(GL_LIGHTING)

        if vertex_format == GL_C4F_N3F_V3F:
            glEnable(GL_COLOR_MATERIAL)

        glInterleavedArrays(vertex_format, 0, material.gl_floats)
        glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

        self.bkt.lock_list[idx].release()

        glPopAttrib()
        glPopClientAttrib()

    def window_load(self, window):
        self.bkt.start()


def main(idx):
    dblist=conf.read_dblist()
    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ClsObj(scene, conf.read_groupset())
        show.show_obj()
        if show.result == 1:
            idx = min(idx + 1, len(dblist) - 1)
        elif show.result == 2:
            idx = max(0, idx - 1)
        else:
            break
        show.bkt.can_exit = True
        show.bkt.join()


if __name__ == '__main__':
    main(0)
