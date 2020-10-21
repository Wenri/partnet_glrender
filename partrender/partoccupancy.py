import faulthandler
from threading import Thread
from time import sleep

import numpy as np
from pyglet.gl import *

from partrender.rendering import RenderObj
from tools.cfgreader import conf


class OccuObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        super().__init__(start_id, not auto_generate, conf.partoccu_dir)
        self.n_samples = 20
        self.cube = None
        self.is_cube_visible = None

    def window_load(self, window):
        self.sample_cube()
        super().window_load(window)

        Thread(target=self, daemon=True).start()

    def sample_cube(self, size=1.0, n=50000):
        self.cube = np.random.uniform(low=-size, high=size, size=[n, 3])
        self.is_cube_visible = np.zeros(shape=(n,), dtype=np.bool)

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
                sleep(1)
            for seed in range(self.n_samples):
                rot_angle = np.random.uniform(low=-180, high=180, size=[2])
                with self.set_render_name(str(seed)):
                    self.rot_angle = np.array(rot_angle, dtype=np.float32)
                sleep(1)

            self.render_ack.wait()
            print(len(self.is_cube_visible) - np.count_nonzero(self.is_cube_visible))

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
    main(0, autogen=True)
