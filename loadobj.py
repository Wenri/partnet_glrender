import os
import numpy as np
from trimesh import load_mesh
from showobj import ShowObj

DATA_DIR = '/media/data/Research/partnet_export'

def main(im_id):
    im_file = os.path.join(DATA_DIR, "{}.obj".format(im_id))
    scene = load_mesh(im_file)

    return
    dim_min = [None, None, None]
    dim_max = [None, None, None]
    for name, v, f in obj_f_v.values():
        for ax in range(len(dim_min)):
            min_v = np.min(v[:, ax])
            max_v = np.max(v[:, ax])
            if not dim_min[ax] or min_v < dim_min[ax][0]:
                dim_min[ax] = (min_v, name)
            if not dim_max[ax] or max_v > dim_max[ax][0]:
                dim_max[ax] = (max_v, name)
    for min_obj, max_obj in zip(dim_min, dim_max):
        print(min_obj, max_obj)
    ShowObj(obj_f_v.values()).show_obj()


if __name__ == '__main__':
    main(753)
