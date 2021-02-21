#!/usr/bin/env python

import faulthandler
import math
import multiprocessing as mp
import os
from threading import Thread

import numpy as np
from pyglet.gl import *

from partrender.occupancy_query import call_and_wait_query_proc
from partrender.partmask import collect_instance_id
from partrender.rendering import RenderObj
from tools.cfgreader import conf

mp.set_start_method('fork')
MAX_N_PTS = 5000000


class OccuObj(RenderObj):
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

    def __init__(self, start_id, auto_generate=False):
        super().__init__(start_id, not auto_generate, conf.partoccu_dir)
        self.n_samples = 20
        self.n_pts_count = None
        self.min_pts_count = None
        self.cube = None
        self.is_cube_visible = None
        self.obj_ins_map = None

    def window_load(self, window):
        super().window_load(window)
        self.obj_ins_map, del_set = collect_instance_id(conf.dblist[self.imageid], self.scene.mesh_list)
        self.del_set.update(del_set)
        self.n_pts_count = 50000
        self.min_pts_count = 16384
        self.sample_cube()

        Thread(target=self, daemon=True).start()

    def sample_cube(self, margin=0.05, global_margin=0.1):
        def sub_sample(sub_vts, sub_margin=margin):
            sub_min = np.min(sub_vts, axis=0) - sub_margin
            sub_max = np.max(sub_vts, axis=0) + sub_margin
            return sub_min, sub_max

        def tri_sample(bound, number):
            sub_p = [np.random.uniform(sub_min, sub_max, size=number) for sub_min, sub_max in zip(bound[0], bound[1])]
            return np.stack(sub_p, axis=-1)

        vts = np.asanyarray(self.scene.vertices, dtype=np.float32)
        sample_p = [
            sub_sample(vts[np.unique(np.asanyarray(mesh.faces, dtype=np.int))]) for mesh in self.scene.mesh_list
        ]
        area = np.array([np.prod(sub_max - sub_min) for sub_min, sub_max in sample_p])
        area_points = area * self.n_pts_count / np.sum(area)
        sample_p = [tri_sample(bound, number) for bound, number in zip(sample_p, area_points.astype(np.int64))]

        sample_p.append(tri_sample(sub_sample(vts, sub_margin=global_margin), self.n_pts_count))

        self.cube = np.concatenate(sample_p, axis=0)
        self.is_cube_visible = np.zeros(shape=(self.cube.shape[0],), dtype=np.bool)

    def perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glGetIntegerv(GL_VIEWPORT, self.viewport)
        glOrtho(-1.2, 1.2, -1.2, 1.2, 0.0, 5.0)

    def save_buffer(self, im_name='render'):
        self.is_cube_visible = np.logical_or(self.is_cube_visible, self.point_is_visible(self.cube))

    def update_cube_visible(self):
        for rot_angle in self.rot_angle_list:
            with self.set_render_name(str(rot_angle)):
                self.rot_angle = np.array(rot_angle, dtype=np.float32)
        for seed in range(self.n_samples):
            rot_angle = np.random.uniform(low=-180, high=180, size=[2])
            with self.set_render_name(str(rot_angle)):
                self.rot_angle = np.array(rot_angle, dtype=np.float32)

        self.render_ack.wait()

    def __call__(self, *args, **kwargs):
        try:
            im_id = conf.dblist[self.imageid]
            print('rendering:', self.imageid, im_id, end=' ')

            # with self.render_lock:
            #     self.update_scene(Wavefront('data/cube.obj', collect_faces=True))
            # sleep(1)

            b_resample = False
            scale_factor = 1
            while True:
                self.update_cube_visible()

                n_cube, _ = self.cube.shape
                n_pts = n_cube - np.count_nonzero(self.is_cube_visible)
                print(f'{n_pts}/{n_cube}', end=' ')

                if n_pts * scale_factor >= self.min_pts_count:
                    break
                elif b_resample:
                    if self.n_pts_count >= MAX_N_PTS:
                        break
                    elif n_pts < 1:
                        self.n_pts_count = MAX_N_PTS
                    else:
                        self.n_pts_count *= int(max(math.ceil(self.min_pts_count / n_pts), 2))
                        if self.n_pts_count > MAX_N_PTS:
                            self.n_pts_count = MAX_N_PTS
                    scale_factor = 2
                    print(f'-> {self.n_pts_count}', end=' ')

                b_resample = True
                self.sample_cube(margin=0.02)

            ins_list = list(self.obj_ins_map.items())

            save_dir = os.path.join(self.render_dir, im_id)
            os.makedirs(save_dir, exist_ok=True)

            call_and_wait_query_proc(
                self.cube, self.is_cube_visible, self.scene.vertices, [mesh.faces for mesh in self.scene.mesh_list],
                ins_list, save_dir
            )

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
    main(5460, autogen=False)  # till 5530
