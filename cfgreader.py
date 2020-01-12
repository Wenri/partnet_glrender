import configparser
import os
from types import SimpleNamespace


class DBReader(SimpleNamespace):
    def __init__(self, configfile):
        cfg = configparser.ConfigParser()
        cfg.read_file(open(configfile))
        self.idname = None
        super().__init__(**cfg.defaults())

    def read_dblist(self):
        with os.scandir(self.data_dir) as it:
            dblist = [os.path.splitext(entry.name)[0] for entry in it
                      if not entry.name.startswith('.') and
                      entry.is_file() and entry.name.endswith('.obj')]
        dblist.sort(key=int)
        return dblist

    def read_idname(self):
        idfile = os.path.join(self.data_dir, 'idname.txt')
        idlist = []
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
                idlist.append((cls.strip(), os.path.basename(file_path)))
                total += 1
        return idlist

    def read_groupset(self):
        grouping_txt = os.path.join(self.data_dir, 'grouping.txt')
        grouping_set = set()
        if not os.path.exists(grouping_txt):
            return grouping_set

        with open(grouping_txt, 'r') as fgrp:
            for line in fgrp:
                line_s = line.strip()
                if line_s:
                    grouping_set.add(line_s)
        return grouping_set

    def mtl_name_to_cls(self, name: str):
        if not self.idname:
            self.idname = self.read_idname()
        prefix, ext = os.path.splitext(name)
        assert prefix == 'Default_OBJ'
        id = int(ext.lstrip('.')) if ext else 0
        return self.idname[id]


conf = DBReader('config.cfg')
