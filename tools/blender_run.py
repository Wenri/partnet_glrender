import sys
from pathlib import Path


def import_file(full_name, path):
    """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    return mod


path = Path(sys.modules[__name__].__file__)
cvt = import_file('cvt', path.parent.joinpath('blender_convert.py'))

for d in 'Bag Bed Bottle Bowl Clock Dishwasher Display Door Earphone Faucet Hat Keyboard Knife Lamp Laptop Microwave Mug Refrigerator Scissors StorageFurniture TrashCan Vase'.split():
    cvt.convert_partnet('/media/data/Research/allexports/{}'.format(d))
