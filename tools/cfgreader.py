import configparser
import os
from collections import defaultdict
from types import SimpleNamespace


class DBReader(SimpleNamespace):
    def __init__(self, configfile):
        cfg = configparser.ConfigParser()
        cfg.read_file(open(configfile))
        self._idname = None
        self._dblist = None
        self._groupset = None
        super().__init__(**cfg.defaults())

    @property
    def dblist(self):
        if self._dblist:
            return self._dblist

        with os.scandir(self.data_dir) as it:
            dblist = [os.path.splitext(entry.name)[0] for entry in it
                      if not entry.name.startswith('.') and
                      entry.is_file() and entry.name.endswith('.obj')]
        dblist.sort(key=int)
        self._dblist = dblist
        return dblist

    @property
    def idname(self):
        if self._idname:
            return self._idname

        idname_txt = os.path.join(self.data_dir, 'idname.txt')
        assert os.path.exists(idname_txt)
        self._idname = defaultdict(list)
        with open(idname_txt) as f:
            for total, line_s in enumerate(s for s in map(str.strip, f) if s):
                id, ins_path, cls, file_path = line_s.split('\t')
                assert total == int(id)
                file_path, im_file = os.path.split(file_path)
                file_path, _ = os.path.split(file_path)
                im_id = os.path.basename(file_path)
                self._idname[im_id].append((ins_path, cls.strip(), im_file))

        return self._idname

    @property
    def groupset(self):
        if self._groupset:
            return self._groupset

        grouping_txt = os.path.join(self.data_dir, 'grouping.txt')
        assert os.path.exists(grouping_txt)
        with open(grouping_txt, 'r') as f:
            self._groupset = set(s for s in map(str.strip, f) if s)

        return self._groupset

    def save_groupset(self, filename):
        with open(filename, mode='w') as f:
            for cls_name in self.groupset:
                print(cls_name, file=f)

    def get_cls_from_mtlname(self, im_id, name):
        prefix, ext = os.path.splitext(name)
        assert prefix == 'Default_OBJ'
        id = int(ext.lstrip('.')) if ext else 0
        im_record = self.idname.get(im_id)
        if not im_record:
            raise ValueError(im_id)
        _, cls, im_file = im_record[id]
        return im_id, cls, im_file

    def find_group_name(self, cls_name):
        while cls_name:
            if cls_name in self.groupset:
                break
            strindex = cls_name.rindex('/')
            cls_name = cls_name[:strindex]
        return cls_name

    def trim_ins_path(self, ins_path, cls_name):
        while cls_name:
            if cls_name in self.groupset:
                return '/'.join(ins_path)
            ins_path.pop()
            strindex = cls_name.rfind('/')
            if strindex > 0:
                cls_name = cls_name[:strindex]
            else:
                raise ValueError(cls_name)
        raise ValueError()


conf = DBReader('config.cfg')
