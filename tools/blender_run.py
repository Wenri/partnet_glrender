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
cvt.convert_partnet('/Volumes/cyber/project/partnet/partnet_blenderexport_tabel')
