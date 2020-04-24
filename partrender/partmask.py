import os
from threading import Thread

import numpy as np
from pywavefront.material import Material

from partrender.rendering import RenderObj
from tools.cfgreader import conf


# The coefficients were taken from OpenCV https://github.com/opencv/opencv
# I'm not sure if the values should be clipped, in my (limited) testing it looks alright
#   but don't hesitate to add rgb.clip(0, 1, rgb) & yuv.clip(0, 1, yuv)
#
# Input for these functions is a numpy array with shape (height, width, 3)
# Change '+= 0.5' to '+= 127.5' & '-= 0.5' to '-= 127.5' for values in range [0, 255]

def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.147108, 0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400, 0.435912, -0.099978]
    ])
    yuv = np.dot(rgb, m)
    yuv[..., 1:] += 0.5
    return yuv


def yuv2rgb(yuv):
    m = np.array([
        [1.000, 1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    yuv[..., 1:] -= 0.5
    rgb = np.dot(yuv, m)
    return rgb


class MaskObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        super(MaskObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)

    def window_load(self, window):
        super(MaskObj, self).window_load(window)
        self.random_seed(self.imageid+1)
        Thread(target=self, daemon=True).start()

    def random_seed(self, seed):
        np.random.seed(seed)

        rx, ry = np.random.random_sample(size=2)
        self.rot_angle = np.array((60*2*(rx-0.5), -30.0*ry), dtype=np.float32)

        u, v = np.random.random_sample(size=2)
        diffuse = yuv2rgb(np.array([0.5, u, v]))
        diffuse = rgb2yuv(np.clip(diffuse, 0, 1))
        for idx, mesh in enumerate(self.scene.mesh_list):
            for material in mesh.materials:
                self.change_mtl(idx, material, diffuse)

    def change_mtl(self, idx, material: Material, diffuse):
        a = np.array(material.vertices, dtype=np.float32).reshape([-1, 6])
        n_vtx, _ = a.shape
        with self.lock_list[idx]:
            color = np.random.standard_normal(size=(n_vtx, 3))
            alpha = np.ones(shape=(n_vtx, 1), dtype=np.float32)
            color *= np.array([0.01, 0.05, 0.05])
            color += diffuse
            color = np.clip(yuv2rgb(color), 0, 1, dtype=np.float32)
            material.gl_floats = np.concatenate((color, alpha, a), axis=1).ctypes
            material.triangle_count = n_vtx
            material.vertex_format = 'C4F_N3F_V3F'

    def __call__(self, *args, **kwargs):
        im_id = conf.dblist[self.imageid]
        try:
            with open(os.path.join(conf.partmask_dir, im_id, 'render-CLSNAME.txt'),
                      mode='w') as f:
                for idx, mesh in enumerate(self.scene.mesh_list):
                    for material in mesh.materials:
                        conf_im_id, cls_name, file_name = conf.get_cls_from_mtlname(material.name)
                        assert conf_im_id == im_id
                        conf_mesh_name, _ = os.path.splitext(file_name)
                        mesh_name, _ = os.path.splitext(mesh.name)
                        assert conf_mesh_name == mesh_name
                        group_name = conf.find_group_name(cls_name)
                        print(conf_im_id, idx, group_name, cls_name, file_name, file=f)
            self.render_ack.wait()

            for i in range(10):
                with self.set_render_name('seed_{}'.format(i), wait=True):
                    v = np.random.randint(0xffffffff)
                    self.random_seed(v)
            self.set_fast_switching()
        except RuntimeError:
            return


def main(idx, autogen=True):
    show = MaskObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
