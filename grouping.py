import os

import io
import json
import tempfile

from urllib.request import urlopen
from urllib.parse import urljoin

#DATA_DIR = 'http://download.cs.stanford.edu/orion/partnet_dataset/data_v0/'
DATA_DIR = 'file:///media/data/Datasets/PartNet/data_v0/'

def load_json(im_id: int):
    d = {}
    for s in ['result', 'result_after_merging', 'meta']:
        with urlopen(urljoin(DATA_DIR, '%d/%s.json' % (im_id, s))) as fp:
            d[s] = json.load(fp)
    return d


def traverse(records, base_name=None):
    for record in records:
        cur_name = base_name + '/' + record['name'] if base_name else record['name']

        if 'children' in record:
            yield from traverse(record['children'], base_name=cur_name)
        elif 'objs' in record:
            yield cur_name, record['objs']
        else:
            raise Exception(cur_name)


def load_obj_files(obj):
    obj_meta = obj['meta']
    obj_dict = set()
    for name, parts in traverse(obj['result_after_merging']):
        for obj_path in parts:
            if obj_path not in obj_dict:
                obj_dict.add(obj_path)
                yield name, urljoin(DATA_DIR, '{}/objs/{}.obj'.format(obj_meta['anno_id'], obj_path))
            else:
                raise Exception(name, obj_path)


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
    parts = load_parts_from_file('/media/data/Research/partnet_export')
    for id, meshobj in enumerate(parts):
        print('%d\t' % id + '\t'.join(meshobj))


if __name__ == '__main__':
    main()
