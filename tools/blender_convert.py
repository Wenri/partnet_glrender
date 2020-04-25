import io
import json
import os
import tempfile
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

DATA_DIR = 'file:///media/data/Datasets/PartNet/data_v0/'


def load_json(im_id: int):
    d = {}
    for s in ['result', 'result_after_merging', 'meta']:
        with urlopen(urljoin(DATA_DIR, '{}/{}.json'.format(im_id, s))) as fp:
            d[s] = json.load(fp)
    return d


def traverse(records, base_name=None, obj_set=None):
    if obj_set is None:
        obj_set = set()
    for record in records:
        cur_name = base_name + '/' + record['name'] if base_name else record['name']

        children_set = set()
        if 'children' in record:
            yield from traverse(record['children'], cur_name, children_set)

        record_objs = record['objs']
        record_set = set(record_objs)
        assert record_set.issuperset(children_set), 'Children have more mesh than parent'
        assert len(record_set) == len(record_objs), 'Node has duplicate mesh'
        assert not record_set.intersection(obj_set), 'Node has duplicate mesh with previous node'
        obj_set.update(record_set)
        record_objs = [o for o in record_objs if o not in children_set]
        if record_objs:
            yield cur_name, record_objs


def load_obj_files(obj):
    obj_meta = obj['meta']
    for name, parts in traverse(obj['result_after_merging']):
        for obj_path in parts:
            yield name, urljoin(DATA_DIR, '{}/objs/{}.obj'.format(obj_meta['anno_id'], obj_path))


def download_id(obj_id):
    obj = load_json(obj_id)
    with tempfile.TemporaryDirectory() as tmpdirname:
        for name, f in load_obj_files(obj):
            parsef = urlparse(f)
            if parsef.scheme == 'file':
                outfile = parsef.path
            else:
                outfile = os.path.join(tmpdirname, os.path.basename(f))
                with urlopen(f) as fp:
                    with open(outfile, 'w') as fout:
                        fout.writelines(io.TextIOWrapper(fp, encoding='utf-8').readlines())
            yield name, outfile


def blender_convert_id(obj_id, save_dir):
    import bpy
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for name, f in download_id(obj_id):
        bpy.ops.import_scene.obj(filepath=f)  # change this line

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.obj(filepath=os.path.join(save_dir, "{}.obj".format(obj_id)))


def main():
    save_dir = '/media/data/Research/partnet_blenderexport'
    list_file = os.path.join(save_dir, 'list.txt')
    with open(list_file) as lstfp:
        for line in lstfp:
            for id in map(int, line.split()):
                blender_convert_id(id, save_dir)
