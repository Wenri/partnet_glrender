def import_file(full_name, path):
    """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    return mod


global cur_id
cvt = import_file('cvt', '/home/wenri/Git/partnet_dataset/blender_render.py')
cvt.main(cur_id)
