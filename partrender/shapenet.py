import faulthandler
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from functools import partial
from itertools import chain
from pathlib import Path
from threading import Thread

import numpy as np
import pcl
from more_itertools import first

from partrender.rendering import RenderObj
from ptcloud.pcmatch import PCMatch
from ptcloud.pointcloud import cvt_obj2pcd
from tools.blender_convert import ShapenetFileHelper
from tools.blender_convert import load_obj_files, load_json
from tools.cfgreader import conf


def acg(start_char, num):
    start_ascii = ord(start_char)
    return (chr(a) for a in range(start_ascii, start_ascii + num))


class ShapeNetObj(RenderObj):
    def __init__(self, start_id, auto_generate=False, load_shapenet=True):
        super(ShapeNetObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)
        self.n_samples = tuple(chain(acg('0', 10), acg('A', 26)))
        self.model_id = 0

    def window_load(self, window):
        super(ShapeNetObj, self).window_load(window)

        self.old_scene = None

        Thread(target=self, daemon=True).start()

    def convert_mesh(self, mesh_list=None):
        if mesh_list is None:
            mesh_list = (mesh for idx, mesh in enumerate(self.scene.mesh_list) if idx not in self.del_set)

        with tempfile.TemporaryDirectory() as d:
            filename = Path(d, conf.dblist[self.imageid]).with_suffix('.obj')
            os.makedirs(filename.parent, exist_ok=True)
            with open(filename, mode='w') as f:
                print("# OBJ file", file=f)
                for v in self.scene.vertices:
                    print("v %.4f %.4f %.4f" % v[:], file=f)
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
                np.save(os.path.join(save_dir, 'render-GT_PC.npy'), self.convert_mesh())
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
    show = ShapeNetObj(idx, autogen, load_shapenet=False)
    show.show_obj()


if __name__ == '__main__':
    main(0, autogen=True)
