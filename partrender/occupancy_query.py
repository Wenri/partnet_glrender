import os
import types
from multiprocessing import Process
from contextlib import closing
from trimesh import Trimesh

import numpy as np

from partrender.numba_proc_aot import query_triangles


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
        with closing(self._get_ids_orig(it, num_results)) as generator:
            return np.fromiter(generator, dtype=np.int64, count=num_results)

    rtree._get_ids_orig = rtree._get_ids
    rtree._get_ids = types.MethodType(_get_ids, rtree)
    # faces that intersect axis aligned bounding box
    candidates = [rtree.intersection(b) for b in bounds]

    return candidates


def get_cube_mtl_id(cube, mask, vertices, mesh_faces):
    n_cube, _ = cube.shape
    mtl_ids = np.zeros(shape=(n_cube,), dtype=np.int)

    if np.count_nonzero(mask) >= n_cube:
        return mtl_ids

    non_visible_mask = np.logical_not(mask)
    pts = cube[non_visible_mask].astype(np.float32)

    all_faces = []
    faces_mtl = []
    for idx, faces in enumerate(mesh_faces):
        all_faces += faces
        faces_mtl += [idx + 1] * len(faces)

    faces_mtl = np.array(faces_mtl, dtype=np.int)
    query = Trimesh(vertices=vertices, faces=all_faces)
    candidates = nearby_faces(query, pts)
    triangles = query.triangles.view(np.ndarray)
    candidates_lens = np.fromiter(map(len, candidates), dtype=np.int64, count=len(candidates))
    print(f'nearby_faces ', np.max(candidates_lens), end=' ')

    candidates = np.concatenate(candidates)
    triangle_id = query_triangles(triangles, candidates, candidates_lens, pts)
    print('query_triangles', end=' ')

    mtl_ids[non_visible_mask] = faces_mtl[triangle_id]

    return mtl_ids


def calc_and_save_results(cube, mask, vertices, faces, ins_list, save_dir):
    mtl_ids = get_cube_mtl_id(cube, mask, vertices, faces)
    in_mask = [mask] + [np.isin(mtl_ids, meshes) for ins_path, meshes in ins_list]

    np.save(os.path.join(save_dir, f'occu-pts.npy'), cube)
    np.save(os.path.join(save_dir, f'occu-ids.npy'), mtl_ids)
    np.save(os.path.join(save_dir, f'occu-isin.npy'), np.argmax(np.stack(in_mask, axis=-1), axis=-1))


def call_and_wait_query_proc(*args):
    p = Process(target=calc_and_save_results, args=args)
    p.start()
    p.join()
