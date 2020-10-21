import faulthandler
import math
import os
from threading import Thread

import numpy as np
from pyglet.gl import *
from trimesh import Trimesh
from trimesh.proximity import closest_point

from partrender.partmask import collect_instance_id
from partrender.rendering import RenderObj
from tools.cfgreader import conf


class OccuObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        super().__init__(start_id, not auto_generate, conf.partoccu_dir)
        self.n_samples = 20
        self.cube = None
        self.is_cube_visible = None
        self.obj_ins_map = None

    def window_load(self, window):
        super().window_load(window)
        self.obj_ins_map, del_set = collect_instance_id(conf.dblist[self.imageid], self.scene.mesh_list)
        self.del_set.update(del_set)
        self.sample_cube()

        Thread(target=self, daemon=True).start()

    def sample_cube(self, n=15000, margin=0.1):
        def sub_sample(sub_vts):
            sub_p = []
            for sub_min, sub_max in zip(np.min(sub_vts, axis=0) - margin, np.max(sub_vts, axis=0) + margin):
                sub_p.append(np.random.uniform(sub_min, sub_max, size=n))
            return np.stack(sub_p, axis=-1)

        vts = np.asanyarray(self.scene.vertices, dtype=np.float32)
        sample_p = [
            sub_sample(vts[np.unique(np.asanyarray(mesh.faces, dtype=np.int))]) for mesh in self.scene.mesh_list
        ]
        sample_p.append(sub_sample(vts))

        self.cube = np.concatenate(sample_p, axis=0)
        self.is_cube_visible = np.zeros(shape=(self.cube.shape[0],), dtype=np.bool)

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glGetIntegerv(GL_VIEWPORT, self.viewport)
        glOrtho(-1.2, 1.2, -1.2, 1.2, 0.0, 5.0)

    def save_buffer(self, im_name='render'):
        self.is_cube_visible = np.logical_or(self.is_cube_visible, self.point_is_visible(self.cube))

    def __call__(self, *args, **kwargs):
        try:
            im_id = conf.dblist[self.imageid]
            print('rendering:', self.imageid, im_id, end=' ')

            # with self.render_lock:
            #     self.update_scene(Wavefront('data/cube.obj', collect_faces=True))
            # sleep(1)

            rot_angle_list = [
                (0, 0), (0, -45), (0, 45), (0, -90), (0, 90),
                (-180, 0), (-180, -45), (-180, 45),

                (-45, 0), (-45, 45), (-45, -45),
                (45, 0), (45, 45), (45, -45),

                (90, 0), (90, 45), (-90, -45),
                (-90, 0), (-90, 45), (-90, -45),

                (-135, 0), (-135, 45), (-135, -45),
                (135, 0), (135, 45), (135, -45)
            ]
            for rot_angle in rot_angle_list:
                with self.set_render_name(str(rot_angle)):
                    self.rot_angle = np.array(rot_angle, dtype=np.float32)
            for seed in range(self.n_samples):
                rot_angle = np.random.uniform(low=-180, high=180, size=[2])
                with self.set_render_name(str(rot_angle)):
                    self.rot_angle = np.array(rot_angle, dtype=np.float32)

            self.render_ack.wait()

            save_dir = os.path.join(self.render_dir, im_id)
            os.makedirs(save_dir, exist_ok=True)
            pts = self.cube[np.logical_not(self.is_cube_visible)].astype(np.float32)
            n_pts, _ = pts.shape
            dist_mtl = []
            print(f'{n_pts}/{len(self.scene.mesh_list)}', end=' ')
            for idx, mesh in enumerate(self.scene.mesh_list):
                print(idx, end=':')
                query = Trimesh(vertices=self.scene.vertices, faces=mesh.faces)
                step = 10000
                dist_all = []
                for i in range(0, n_pts, step):
                    _, dist, _ = closest_point(query, pts[i:i + step, :])
                    dist_all.append(dist)
                    progress = math.floor(10 * (i + step) / n_pts)
                    print('D' if progress >= 10 else progress, end='')
                print(' ', end='')
                dist_mtl.append(np.concatenate(dist_all, axis=0))

            mtl_ids = np.argmin(np.stack(dist_mtl, axis=-1), axis=-1) + 1
            ins_list = list(self.obj_ins_map.items())
            in_mask = np.stack([np.isin(mtl_ids, meshes) for ins_path, meshes in ins_list], axis=-1)

            np.save(os.path.join(save_dir, f'occu-pts.npy'), pts)
            np.save(os.path.join(save_dir, f'occu-ids.npy'), mtl_ids)
            np.save(os.path.join(save_dir, f'occu-isin.npy'), np.argmax(in_mask, axis=-1))

            if not self.view_mode:
                print('Switching...')
                self.set_fast_switching()
            else:
                self.render_ack.wait()
                print('Done.')

        except RuntimeError:
            return


def main(idx, autogen=True):
    faulthandler.enable()
    show = OccuObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(311, autogen=True)
