import configparser
import os
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

        idfile = os.path.join(self.data_dir, 'idname.txt')
        idlist = []
        self._idname = idlist
        if not os.path.exists(idfile):
            return idlist

        total = 0
        with open(idfile) as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                id, cls, file_path = line_s.split('\t')
                assert total == int(id)
                file_path, im_file = os.path.split(file_path)
                file_path, _ = os.path.split(file_path)
                idlist.append((os.path.basename(file_path), cls.strip(), im_file))
                total += 1
        return idlist

    @property
    def groupset(self):
        if self._groupset:
            return self._groupset

        grouping_txt = os.path.join(self.data_dir, 'grouping.txt')
        grouping_set = set()
        self._groupset = grouping_set
        if not os.path.exists(grouping_txt):
            return grouping_set

        with open(grouping_txt, 'r') as fgrp:
            for line in fgrp:
                line_s = line.strip()
                if line_s:
                    grouping_set.add(line_s)
        return grouping_set

    def get_cls_from_mtlname(self, name: str):
        prefix, ext = os.path.splitext(name)
        assert prefix == 'Default_OBJ'
        id = int(ext.lstrip('.')) if ext else 0
        return self.idname[id]

    def find_group_name(self, cls_name: str):
        while cls_name:
            if cls_name in self.groupset:
                break
            strindex = cls_name.rindex('/')
            cls_name = cls_name[:strindex]
        return cls_name


conf = DBReader('config.cfg')
