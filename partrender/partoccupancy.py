import faulthandler
import math
import os
import types
from threading import Thread

import numpy as np
from numba import njit
from pyglet.gl import *
from trimesh import Trimesh

from partrender.partmask import collect_instance_id
from partrender.rendering import RenderObj
from tools.cfgreader import conf


def nearby_faces(mesh, points, tol_merge=1e-8):
    # an r-tree containing the axis aligned bounding box for every triangle
    rtree = mesh.triangles_tree
    # a kd-tree containing every vertex of the mesh
    kdtree = mesh.kdtree

    # query the distance to the nearest vertex to get AABB of a sphere
    distance_vertex = kdtree.query(points)[0].reshape((-1, 1))
    distance_vertex += tol_merge

    # axis aligned bounds
    bounds = np.column_stack((points - distance_vertex,
                              points + distance_vertex))

    def _get_ids(self, it, num_results):
        return np.fromiter(self._get_ids_orig(it, num_results), dtype=np.int64, count=num_results)

    rtree._get_ids_orig = rtree._get_ids
    rtree._get_ids = types.MethodType(_get_ids, rtree)
    # faces that intersect axis aligned bounding box
    candidates = [rtree.intersection(b) for b in bounds]

    return candidates


@njit(fastmath=True)
def query_triangles(all_triangles, candidates, pts):
    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = np.ones(3)
    tol_zero = np.finfo(np.float64).resolution * 100

    def closest_point_corresponding(triangles, points):
        # check input triangles and points
        n_triangles, _, _ = triangles.shape

        # store the location of the closest point
        result = np.zeros((n_triangles, 3))
        # which points still need to be handled
        remain = np.ones(n_triangles, dtype=np.bool_)

        # get the three points of each triangle
        # use the same notation as RTCD to avoid confusion
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]

        # check if P is in vertex region outside A
        ab = b - a
        ac = c - a
        ap = points - a
        # this is a faster equivalent of:
        # diagonal_dot(ab, ap)
        d1 = np.dot(ab * ap, ones)
        d2 = np.dot(ac * ap, ones)

        # is the point at A
        is_a = np.logical_and(d1 < tol_zero, d2 < tol_zero)
        if np.any(is_a):
            result[is_a] = a[is_a]
            remain[is_a] = False

        # check if P in vertex region outside B
        bp = points - b
        d3 = np.dot(ab * bp, ones)
        d4 = np.dot(ac * bp, ones)

        # do the logic check
        is_b = (d3 > -tol_zero) & (d4 <= d3) & remain
        if np.any(is_b):
            result[is_b] = b[is_b]
            remain[is_b] = False

        # check if P in edge region of AB, if so return projection of P onto A
        vc = (d1 * d4) - (d3 * d2)
        is_ab = ((vc < tol_zero) &
                 (d1 > -tol_zero) &
                 (d3 < tol_zero) & remain)
        if np.any(is_ab):
            v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
            result[is_ab] = a[is_ab] + (v * ab[is_ab])
            remain[is_ab] = False

        # check if P in vertex region outside C
        cp = points - c
        d5 = np.dot(ab * cp, ones)
        d6 = np.dot(ac * cp, ones)
        is_c = (d6 > -tol_zero) & (d5 <= d6) & remain
        if np.any(is_c):
            result[is_c] = c[is_c]
            remain[is_c] = False

        # check if P in edge region of AC, if so return projection of P onto AC
        vb = (d5 * d2) - (d1 * d6)
        is_ac = (vb < tol_zero) & (d2 > -tol_zero) & (d6 < tol_zero) & remain
        if np.any(is_ac):
            w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
            result[is_ac] = a[is_ac] + w * ac[is_ac]
            remain[is_ac] = False

        # check if P in edge region of BC, if so return projection of P onto BC
        va = (d3 * d6) - (d5 * d4)
        is_bc = ((va < tol_zero) &
                 ((d4 - d3) > - tol_zero) &
                 ((d5 - d6) > -tol_zero) & remain)
        if np.any(is_bc):
            d43 = d4[is_bc] - d3[is_bc]
            w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
            result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
            remain[is_bc] = False

        # any remaining points must be inside face region
        if np.any(remain):
            # point is inside face region
            denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
            v = (vb[remain] * denom).reshape((-1, 1))
            w = (vc[remain] * denom).reshape((-1, 1))
            # compute Q through its barycentric coordinates
            result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

        return result

    n_pts, _ = pts.shape
    ret = np.empty(n_pts, dtype=np.int64)
    for i, fid in enumerate(candidates):
        qv = closest_point_corresponding(all_triangles[fid], pts[i]) - pts[i]
        ret[i] = fid[np.argmin(np.dot(qv * qv, ones))]
    return ret


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

    def sample_cube(self, margin=0.05):
        def sub_sample(sub_vts):
            sub_min = np.min(sub_vts, axis=0) - margin
            sub_max = np.max(sub_vts, axis=0) + margin
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

        sample_p.append(tri_sample(sub_sample(vts), self.n_pts_count))

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

    def get_cube_mtl_id(self):
        n_cube, _ = self.cube.shape
        mtl_ids = np.zeros(shape=(n_cube,), dtype=np.int)

        if np.count_nonzero(self.is_cube_visible) >= n_cube:
            return mtl_ids

        non_visible_mask = np.logical_not(self.is_cube_visible)
        pts = self.cube[non_visible_mask].astype(np.float32)

        all_faces = []
        faces_mtl = []
        for idx, mesh in enumerate(self.scene.mesh_list):
            all_faces += mesh.faces
            faces_mtl += [idx + 1] * len(mesh.faces)

        faces_mtl = np.array(faces_mtl, dtype=np.int)
        query = Trimesh(vertices=self.scene.vertices, faces=all_faces)
        candidates = nearby_faces(query, pts)
        triangles = query.triangles.view(np.ndarray)
        candidates_lens = np.fromiter(map(len, candidates), dtype=np.int64, count=len(candidates))
        print(f'nearby_faces ', np.max(candidates_lens), end=' ')

        # candidates = np.concatenate(candidates)
        # print('np_concatenate', end=' ')

        triangle_id = query_triangles(triangles, candidates, pts)
        print('query_triangles', end=' ')

        mtl_ids[non_visible_mask] = faces_mtl[triangle_id]

        return mtl_ids

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

                save_dir = os.path.join(self.render_dir, im_id)
                os.makedirs(save_dir, exist_ok=True)
                n_cube, _ = self.cube.shape
                n_pts = n_cube - np.count_nonzero(self.is_cube_visible)
                print(f'{n_pts}/{n_cube}', end=' ')

                if n_pts * scale_factor >= self.min_pts_count:
                    break
                elif b_resample:
                    if n_pts < 10:
                        break
                    self.n_pts_count *= int(max(math.ceil(self.min_pts_count / n_pts), 2))
                    scale_factor = 2
                    print(f'-> {self.n_pts_count}', end=' ')

                b_resample = True
                self.sample_cube(margin=0.02)

            mtl_ids = self.get_cube_mtl_id()
            ins_list = list(self.obj_ins_map.items())
            in_mask = [self.is_cube_visible] + [np.isin(mtl_ids, meshes) for ins_path, meshes in ins_list]

            np.save(os.path.join(save_dir, f'occu-pts.npy'), self.cube)
            np.save(os.path.join(save_dir, f'occu-ids.npy'), mtl_ids)
            np.save(os.path.join(save_dir, f'occu-isin.npy'), np.argmax(np.stack(in_mask, axis=-1), axis=-1))

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
    main(567, autogen=True)
