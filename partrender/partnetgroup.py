import os
from itertools import chain

from tools import blender_convert
from tools.blender_convert import load_json, load_obj_files
from tools.cfgreader import conf

blender_convert.DATA_URL = conf.partnet_url


def load_parts_from_file(save_dir):
    list_file = os.path.join(save_dir, 'list.txt')
    with open(list_file) as lstfp:
        for id in chain.from_iterable(line.split() for line in lstfp):
            yield from load_obj_files(load_json(int(id)))


def main():
    parts = load_parts_from_file(conf.data_dir)
    for id, meshobj in enumerate(parts):
        print('%d\t' % id + '\t'.join(meshobj))


if __name__ == '__main__':
    main()
