import io
import json
import numpy as np
from urllib.request import urlopen, urljoin
from geometry_utils import load_obj
from showobj import ShowObj

DATA_DIR = 'http://download.cs.stanford.edu/orion/partnet_dataset/data_v0/'


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


def load_obj_f_v(obj):
    obj_meta = obj['meta']
    obj_dict = {}
    for name, parts in traverse(obj['result_after_merging']):
        for obj_path in parts:
            if obj_path not in obj_dict:
                with urlopen(urljoin(DATA_DIR, '{}/objs/{}.obj'.format(obj_meta['anno_id'], obj_path))) as fp:
                    obj_dict[obj_path] = (name,) + load_obj(io.TextIOWrapper(fp, encoding='utf-8'))
            else:
                raise Exception(name, obj_path)
    return obj_dict


def main(im_id):
    obj = load_json(im_id)
    obj_f_v = load_obj_f_v(obj)
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
