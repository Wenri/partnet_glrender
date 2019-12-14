import os
import numpy as np
from pywavefront import Wavefront
from showobj import ShowObj
from dblist import dblist, conf

def main(idx):

    while True:
        im_id = dblist[idx]
        im_file = os.path.join(conf.data_dir, "{}.obj".format(im_id))
        scene = Wavefront(im_file)
        show = ShowObj(scene)
        show.show_obj()
        if show.result == 1:
            idx = min(idx+1, len(dblist)-1)
        elif show.result == 2:
            idx = max(0, idx-1)
        else:
            break


if __name__ == '__main__':
    main(0)
