import glob
import io
import json
import os
import tempfile
from contextlib import ExitStack
from itertools import chain
from operator import itemgetter
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

DATA_URL = 'file:///Volumes/gbcdisk/research/PartNet/data_v0/'
SHAPENET_PREFIX = '/Volumes/gbcdisk/research/ShapeNet/'


def path_split_all(path):
    path, part = os.path.split(path)
    if path:
        yield from path_split_all(path)
    yield part


def load_json(obj_id):
    def _load_s(s):
        with urlopen(urljoin(DATA_URL, '{}/{}.json'.format(obj_id, s))) as fp:
            return json.load(fp)

    return SimpleNamespace(**{k: _load_s(k) for k in ('result', 'result_after_merging', 'meta')})


def traverse(records, base_name=None, obj_set=None):
    if obj_set is None:
        obj_set = set()
    for record in records:
        cur_name = base_name + '/' + record['name'] if base_name else record['name']

        record_objs = record['objs']
        record_set = set(record_objs)
        assert len(record_set) == len(record_objs) and not record_set.intersection(obj_set), 'Node has duplicate mesh'
        obj_set.update(record_set)

        if 'children' in record:
            children_set = set()
            yield from traverse(record['children'], cur_name, children_set)
            assert record_set.issuperset(children_set), 'Children have more mesh than parent'
            record_objs = [o for o in record_objs if o not in children_set]

        if record_objs:
            yield cur_name, record_objs


def load_obj_files(obj):
    for name, parts in traverse(obj.result_after_merging):
        for obj_path in parts:
            yield name, urljoin(DATA_URL, '{}/objs/{}.obj'.format(obj.meta['anno_id'], obj_path))


def download_id(obj_id):
    tmpdirname = None
    with ExitStack() as stack:
        for name, f in load_obj_files(load_json(obj_id)):
            parsef = urlparse(f)
            if parsef.scheme == 'file':
                outfile = parsef.path
            else:
                if tmpdirname is None:
                    tmpdirname = stack.enter_context(tempfile.TemporaryDirectory())
                outfile = os.path.join(tmpdirname, os.path.basename(f))
                with urlopen(f) as fp, open(outfile, 'w') as fout:
                    fout.writelines(io.TextIOWrapper(fp, encoding='utf-8').readlines())
            yield name, outfile


class ShapenetFileHelper:
    def __init__(self, prefix, version=('v2', 'v1'), pattern=os.path.join('shapenetcore{v}', 'ShapeNetCore.{v}')):
        def _get_taxonomy(d):
            with open(os.path.join(d, 'taxonomy.json')) as f:
                return [SimpleNamespace(**a) for a in json.load(f)]

        if isinstance(version, str):
            version = [version]

        self.shapenet_dir = [os.path.join(prefix, pattern.format(v=v)) for v in version]
        self.taxonomy = [_get_taxonomy(d) for d in self.shapenet_dir]

    def _gen_synset_id(self, db_id, cat=None):
        for a in self.taxonomy[db_id]:
            if not cat or cat in a.name.lower():
                yield a.synsetId
                yield from a.children

    def _get_synset_id_set(self, db_id, cat_hint):
        cat_s = set(self._gen_synset_id(db_id, cat_hint.lower()))
        return cat_s, set(self._gen_synset_id(db_id)) - cat_s

    def find_model_id(self, obj_id):
        model_id, model_cat = itemgetter('model_id', 'model_cat')(load_json(obj_id).meta)
        for db_id in range(len(self.shapenet_dir)):
            for synset_id in chain.from_iterable(self._get_synset_id_set(db_id, model_cat)):
                if os.path.exists(synset_path := os.path.join(self.shapenet_dir[db_id], synset_id, model_id)):
                    yield synset_path

    def __call__(self, obj_id, all_db=False):
        for synset_path in self.find_model_id(obj_id):
            print('Hit {}'.format(synset_path))
            for f in glob.glob(os.path.join(synset_path, '**', '*.obj'), recursive=True):
                name = os.path.basename(f)
                if name.startswith('model'):
                    yield name, f
            if not all_db:
                return

        raise RuntimeError("obj_id not found: {}".format(obj_id))


def blender_convert_id(obj_id, save_dir, helper):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print('Downloading obj_id {}'.format(obj_id))
    for name, f in helper(obj_id):
        bpy.ops.import_scene.obj(filepath=f)  # change this line

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.obj(filepath=os.path.join(save_dir, "{}.obj".format(obj_id)))


def convert_partnet(save_dir):
    list_file = os.path.join(save_dir, 'list.txt')
    with open(list_file) as lstfp:
        for id in chain.from_iterable(line.split() for line in lstfp):
            blender_convert_id(int(id), save_dir, helper=download_id)


def convert_shapenet(save_dir, start=0):
    list_file = os.path.join(save_dir, 'list.txt')
    with open(list_file) as lstfp:
        for i, id in enumerate(chain.from_iterable(line.split() for line in lstfp)):
            if i < start:
                continue
            blender_convert_id(int(id), save_dir, helper=ShapenetFileHelper(SHAPENET_PREFIX, version='v1'))
