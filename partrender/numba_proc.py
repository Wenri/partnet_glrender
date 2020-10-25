import numpy as np
from numba.pycc import CC

cc = CC('numba_proc_aot')
cc.target_cpu = 'host'
cc.verbose = True


# readonly array(float64, 3d, C) reflected list(array(int64, 1d, C))<iv=None> array(float32, 2d, C)
@cc.export('query_triangles', '(f8[:, :, :], i8[:], i8[:], f4[:,:])')
def query_triangles(all_triangles, candidates, candidates_lens, pts):
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
    cur_start = 0
    for i, cur_len in enumerate(candidates_lens):
        fid = candidates[cur_start:cur_start + cur_len]
        qv = closest_point_corresponding(all_triangles[fid], pts[i]) - pts[i]
        ret[i] = fid[np.argmin(np.dot(qv * qv, ones))]
        cur_start += cur_len
    return ret


if __name__ == "__main__":
    cc.compile()
