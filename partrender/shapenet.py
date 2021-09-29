import faulthandler
import os
import tempfile
from collections import namedtuple
from itertools import chain
from pathlib import Path
from threading import Thread

import numpy as np

from partrender.rendering import RenderObj, get_bbox, acg
from tools.cfgreader import conf


class ShapeNetObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        super(ShapeNetObj, self).__init__(start_id, not auto_generate, conf.partoccu_dir)
        self.n_samples = tuple(chain(acg('0', 10), acg('A', 26)))
        self.model_id = 0
        self.target_pc = None
        self.target_occ = None
        self.scale_inv = None
        self.trans_neg = None

    def load_shapenet_pc(self, filename='pointcloud.npz', fields=None):
        im_id = conf.dblist[self.imageid]
        npz_file = Path(conf.partoccu_dir, im_id).parent / filename
        npz = np.load(npz_file)
        if fields is None:
            fields = tuple(npz)
        shapenet_pc = namedtuple(npz_file.with_suffix('').name, fields)
        return shapenet_pc._make(npz[a] for a in shapenet_pc._fields)

    def window_load(self, window):
        self.target_pc = self.load_shapenet_pc()
        self.target_occ = self.load_shapenet_pc(filename='points.npz')
        self.scale_inv = np.ones(3) / self.target_pc.scale
        self.trans_neg = -self.target_pc.loc

        super(ShapeNetObj, self).window_load(window)

        bb_min, bb_max = get_bbox(np.asarray(self.scene.vertices))
        padding = 0.0

        bb_min, bb_max = np.array(bb_min), np.array(bb_max)
        total_size = (bb_max - bb_min).max()
        # Set the center (although this should usually be the origin already).
        centers = (bb_min + bb_max) / 2

        # Scales all dimensions equally.
        scale = total_size / (1 - padding)
        assert np.allclose(scale, self.target_pc.scale) and np.allclose(centers, self.target_pc.loc)
        assert np.allclose(scale, self.target_occ.scale) and np.allclose(centers, self.target_occ.loc)

        Thread(target=self, daemon=True).start()

    def draw_model(self):
        with self.matrix_trans(translate=self.trans_neg, scale=self.scale_inv):
            super(ShapeNetObj, self).draw_model()

    def convert_mesh(self, mesh_list=None):
        if mesh_list is None:
            mesh_list = (mesh for idx, mesh in enumerate(self.scene.mesh_list) if idx not in self.del_set)

        with tempfile.TemporaryDirectory() as d:
            filename = Path(d, conf.dblist[self.imageid]).with_suffix('.obj')
            os.makedirs(filename.parent, exist_ok=True)
            vertices = np.asarray(self.scene.vertices)
            vertices += self.trans_neg
            vertices *= self.scale_inv
            with open(filename, mode='w') as f:
                print("# OBJ file", file=f)
                for v in vertices:
                    print("v %.4f %.4f %.4f" % tuple(v[:3]), file=f)
                for m in mesh_list:
                    for p in m.faces:
                        print("f %d %d %d" % tuple(i + 1 for i in p), file=f)
            return np.array(self.load_pcd(d, leaf_size=0.0015))

    def __call__(self, *args, **kwargs):
        try:
            im_id = conf.dblist[self.imageid]
            print('rendering:', self.imageid, im_id, self.model_id, end=' ')

            if not self.view_mode:
                save_dir = os.path.join(self.render_dir, im_id)
                os.makedirs(save_dir, exist_ok=True)
                this_pc = self.convert_mesh()
                # bb_min0, bb_max0 = get_bbox(self.target_pc.points)
                # bb_min, bb_max = get_bbox(this_pc)
                np.save(os.path.join(save_dir, 'render-GT_PC.npy'), this_pc)
                for sn in self.n_samples:
                    with self.set_render_name('seed_{}'.format(sn), wait=True):
                        self.random_seed('{}-{}'.format(self.imageid, sn))

                print('Switching...')
                self.set_fast_switching()
            else:
                self.render_ack.wait()
                print('Done.')

        except RuntimeError:
            return


def main(idx, autogen=True):
    faulthandler.enable()
    conf.cfg_def_str = 'SHAPENET'
    show = ShapeNetObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0, autogen=True)
