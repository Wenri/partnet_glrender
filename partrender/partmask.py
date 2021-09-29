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
from more_itertools import first

from partrender.rendering import RenderObj, acg
from ptcloud.pcmatch import PCMatch
from tools.blender_convert import ShapenetFileHelper
from tools.blender_convert import load_obj_files, load_json
from tools.cfgreader import conf


def collect_instance_id(im_id, mesh_list):
    meta = load_json(im_id)

    rlookup = {os.path.splitext(mesh.name)[0]: idx + 1 for idx, mesh in enumerate(mesh_list)}
    obj_ins_map = defaultdict(list)
    del_set = set()
    for ins_path, cls_name, obj in load_obj_files(meta):
        obj_name = os.path.splitext(os.path.basename(obj))[0]
        try:
            ins_id = conf.trim_ins_path(ins_path.split('/'), cls_name)
            obj_ins_map[ins_id].append(rlookup[obj_name])
        except ValueError as e:
            print('Skipping {} due to {}'.format(obj_name, e), file=sys.stderr)
            del_set.add(rlookup[obj_name] - 1)

    return obj_ins_map, del_set


class MaskObj(RenderObj):
    def __init__(self, start_id, auto_generate=False, load_shapenet=True):
        super(MaskObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)
        self.n_samples = tuple(chain(acg('0', 10), acg('A', 26)))
        self.matched_matrix = None
        self.should_apply_trans = False
        self.should_load_shapenet = load_shapenet
        self.old_scene = None
        self.model_id = 0
        self.obj_ins_map = None

        self.act_key('T', self.swap_scene)
        self.act_key('G', partial(self.swap_scene, toggle_trans=False))

        if auto_generate:
            conf.save_groupset(os.path.join(conf.partmask_dir, 'grouping.txt'))

    def swap_scene(self, toggle_trans=True):
        if self.old_scene:
            self.old_scene = self.update_scene(self.old_scene)
            if toggle_trans:
                self.should_apply_trans = not self.should_apply_trans

    def calc_apply_matched_matrix(self):
        partnet_pcd = self.load_pcd(conf.data_dir, leaf_size=0.002)
        shapenet_pcd = self.load_pcd(conf.shapenet_dir, leaf_size=0.001)
        pcm = PCMatch(partnet_pcd, shapenet_pcd)
        sim_min, trans, offset = pcm.rotmatrix_match()
        trans_m = np.hstack((np.transpose(trans), offset[:, np.newaxis]))
        self.matched_matrix = np.vstack((trans_m, np.array([[0., 0., 0., 1.]])))
        self.should_apply_trans = True

    def load_shapenet(self):
        helper = ShapenetFileHelper(prefix=conf.shapenet_prefix)
        search_path = first(helper.find_model_id(conf.dblist[self.imageid]))
        print(f"{search_path=}")
        self.model_id = os.path.basename(search_path)
        scene = self.load_image(conf.shapenet_dir)
        for m in chain.from_iterable(mesh.materials for mesh in scene.mesh_list):
            if t := m.texture:
                t._search_path = Path(search_path)
        return scene

    def window_load(self, window):
        super(MaskObj, self).window_load(window)

        self.old_scene = None
        if self.should_load_shapenet:
            try:
                scene = self.load_shapenet()
                self.calc_apply_matched_matrix()
                self.old_scene = self.update_scene(scene)
            except subprocess.CalledProcessError as e:
                print(first(e.cmd), 'err code', e.returncode, file=sys.stderr)
            except (FileNotFoundError, IOError) as e:
                print(e, file=sys.stderr)
        self.obj_ins_map, del_set = collect_instance_id(conf.dblist[self.imageid], self.scene.mesh_list)
        self.del_set.update(del_set)

        Thread(target=self, daemon=True).start()

    def draw_model(self):
        with ExitStack() as stack:
            if self.should_apply_trans:
                stack.enter_context(self.matrix_trans(self.matched_matrix))
            super(MaskObj, self).draw_model()

    def convert_mesh(self, mesh_list=None):
        if mesh_list is None:
            mesh_list = (mesh for idx, mesh in enumerate(self.scene.mesh_list) if idx not in self.del_set)

        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, conf.dblist[self.imageid] + '.obj')
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
                scene = self.old_scene if self.old_scene else self.scene
                with open(os.path.join(save_dir, 'render-CLSNAME.txt'), mode='w') as f:
                    for idx, mesh in enumerate(scene.mesh_list):
                        for material in mesh.materials:
                            conf_im_id, cls_name, file_name = conf.get_cls_from_mtlname(im_id, material.name)
                            assert conf_im_id == im_id
                            conf_mesh_name, _ = os.path.splitext(file_name)
                            mesh_name, _ = os.path.splitext(mesh.name)
                            assert conf_mesh_name == mesh_name
                            print(conf_im_id, idx, cls_name, file_name, file=f)
                ins_list = list(self.obj_ins_map.items())
                with open(os.path.join(save_dir, 'render-INSNAME.txt'), mode='w') as f:
                    for ins_path, meshes in ins_list:
                        print(ins_path, ','.join(map(str, meshes)), file=f)
                np.save(os.path.join(save_dir, 'render-GT_PC.npy'), self.convert_mesh())
                # ins_pc = [self.convert_mesh([self.scene.mesh_list[idx - 1] for idx in meshes])
                #           for _, meshes in ins_list]
                # np.save(os.path.join(save_dir, 'render-INS_PC.npy'), np.array(ins_pc, dtype=np.object))
                for sn in self.n_samples:
                    with self.set_render_name('seed_{}'.format(sn), wait=True):
                        self.swap_scene()
                        self.random_seed('{}-{}'.format(self.imageid, sn))
                    if self.old_scene:
                        with self.set_render_name('seed_{}T'.format(sn), wait=True):
                            self.swap_scene()

                print('Switching...')
                self.set_fast_switching()
            else:
                self.render_ack.wait()
                print('Done.')

        except RuntimeError:
            return


def main(idx, autogen=True):
    faulthandler.enable()
    show = MaskObj(idx, autogen, load_shapenet=False)
    show.show_obj()


if __name__ == '__main__':
    main(0, autogen=True)
