import os
from pywavefront import Wavefront
from dblist import dblist, conf
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


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


def main(idx):
    im_id = dblist[idx]
    im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
    scene = Wavefront(im_file)
    for mtl in scene.materials.values():
        normal_cls_functor(mtl.vertex_format)(mtl.vertices)


if __name__ == '__main__':
    main(2)
