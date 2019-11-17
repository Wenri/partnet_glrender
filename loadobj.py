import os
import numpy as np
from pywavefront import Wavefront
from showobj import ShowObj

DATA_DIR = '/media/data/Research/partnet_export'
#DATA_DIR = '/Users/wenri/Research/partnet_export'


def main(im_id):
    im_file = os.path.join(DATA_DIR, "{}.obj".format(im_id))
    scene = Wavefront(im_file)
    
    ShowObj(scene).show_obj()


if __name__ == '__main__':
    main(753)
