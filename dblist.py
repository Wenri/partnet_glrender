import configparser
import os
from types import SimpleNamespace

cfg = configparser.ConfigParser()
cfg.read_file(open('config.cfg'))
conf = SimpleNamespace(**cfg.defaults())


def init_dblist():
    with os.scandir(conf.data_dir) as it:
        dblist = [os.path.splitext(entry.name)[0] for entry in it
                  if not entry.name.startswith('.') and
                  entry.is_file() and entry.name.endswith('.obj')]
    dblist.sort(key=int)
    return dblist


def init_idname():
    idfile = os.path.join(conf.data_dir, 'idname.txt')
    if not os.path.exists(idfile):
        return []
    idlist = []
    total = 0
    with open(idfile) as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            id, cls, file_path = line_s.split('\t')
            assert total == int(id)
            idlist.append((cls, os.path.basename(file_path)))
            total += 1
    return idlist

dblist = init_dblist()
idname = init_idname()
