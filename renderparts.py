import os
from pywavefront import Wavefront
from pywavefront.mesh import Mesh
from dblist import dblist, conf
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from showobj import ShowObj
from threading import Thread


def get_normal_cls(vertices):
    a = np.array(vertices).reshape([-1, 6])
    vtx, vn = a[:, :3], a[:, 3:]

    sim_mat = np.abs(cosine_similarity(vn))
    clustering = SpectralClustering(n_clusters=3,
                                    affinity='precomputed')
    cls = clustering.fit_predict(sim_mat + np.finfo(np.float32).eps)
    ret = [np.count_nonzero(cls == i) for i in range(3)]
    ret.sort(reverse=True)

    print('\t'.join(str(s) for s in ret))

    ts = TSNE(metric='precomputed')
    vis = ts.fit_transform(1 - sim_mat + np.finfo(np.float32).eps)
    plt.figure()
    plt.scatter(vis[:, 0], vis[:, 1], c=cls)
    plt.show()

    for i in range(3):
        print(np.mean(vn[cls == i], axis=0))


def normal_cls_functor(pixel_format):
    pixel_normal = {
        'N3F_V3F': get_normal_cls
    }
    return pixel_normal[pixel_format]


def analysis(mesh: Mesh):
    print('analysis mesh: %s' % mesh.name)
    for mtl in mesh.materials:
        normal_cls_functor(mtl.vertex_format)(mtl.vertices)


def main(idx):
    class ClsObj(ShowObj):
        def do_part(self, partid):
            mesh = self.scene.mesh_list[partid]
            t = Thread(target=analysis, args=(mesh,), daemon=True)
            t.start()

    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ClsObj(scene)
        show.show_obj()
        if show.result == 1:
            idx = min(idx+1, len(dblist)-1)
        elif show.result == 2:
            idx = max(0, idx-1)
        else:
            break


if __name__ == '__main__':
    main(0)
