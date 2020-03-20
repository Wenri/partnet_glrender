import os

import blender_convert
from blender_convert import load_json, load_obj_files
from cfgreader import conf

blender_convert.DATA_DIR = conf.partnet_url


def load_obj_from_id(obj_id, save_dir):
    obj = load_json(obj_id)
    yield from load_obj_files(obj)


def load_parts_from_file(save_dir):
    list_file = os.path.join(save_dir, 'list.txt')
    with open(list_file) as lstfp:
        for line in lstfp:
            for id in map(int, line.split()):
                yield from load_obj_from_id(id, save_dir)


def main():
    parts = load_parts_from_file(conf.data_dir)
    for id, meshobj in enumerate(parts):
        print('%d\t' % id + '\t'.join(meshobj))


if __name__ == '__main__':
    main()
