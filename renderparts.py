import os
import time

from pywavefront import Wavefront
from pywavefront.mesh import Mesh
from dblist import dblist, conf
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import spectral_embedding
from sklearn.cluster import k_means
from matplotlib import pyplot as plt
from showobj import ShowObj
from threading import Thread
from pyglet.gl import *
from pywavefront.visualization import gl_light


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


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

    def do_part(self, partid):
        mesh = self.scene.mesh_list[partid]
        t = Thread(target=analysis, args=(mesh,), daemon=True)
        t.start()

    def draw_material(self, material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
        """Draw a single material"""
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

        glInterleavedArrays(vertex_format, 0, material.gl_floats)
        glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

        glPopAttrib()
        glPopClientAttrib()


def get_normal_cls(vertices):
    a = np.array(vertices).reshape([-1, 6])
    nvtx, _ = a.shape
    print("Vtx Count: %d" % nvtx)

    n9f = np.reshape(normalize(a[:, :3]), (-1, 3, 3))
    n3f = np.mean(n9f, axis=1)

    with Timer("abs_cosine_similarity"):
        sim_mat = np.abs(cosine_similarity(n3f))
    with Timer("spectral_embedding"):
        embedding = spectral_embedding(sim_mat, n_components=3, drop_first=False)
    with Timer("k_means"):
        _, labels, _ = k_means(embedding, n_clusters=3)

    ret = [np.count_nonzero(labels == i) for i in range(3)]
    ret.sort(reverse=True)

    print('\t'.join(str(s) for s in ret))

    plt.figure()
    plt.scatter(embedding[:, 1], embedding[:, 2], c=labels)
    plt.show()


def normal_cls_functor(pixel_format):
    pixel_normal = {
        'N3F_V3F': get_normal_cls
    }
    return pixel_normal[pixel_format]


def analysis(mesh: Mesh):
    print('analysis mesh: %s' % mesh.name)
    for mtl in mesh.materials:
        normal_cls_functor(mtl.vertex_format)(mtl.vertices)


def main(idx):
    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ClsObj(scene)
        show.show_obj()
        if show.result == 1:
            idx = min(idx + 1, len(dblist) - 1)
        elif show.result == 2:
            idx = max(0, idx - 1)
        else:
            break


if __name__ == '__main__':
    main(0)
