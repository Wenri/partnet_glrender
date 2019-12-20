import configparser
import os
from types import SimpleNamespace

cfg = configparser.ConfigParser()
cfg.read_file(open('config.cfg'))
conf = SimpleNamespace(**cfg.defaults())

with os.scandir(conf.data_dir) as it:
    dblist = [os.path.splitext(entry.name)[0] for entry in it
              if not entry.name.startswith('.') and
              entry.is_file() and entry.name.endswith('.obj')]

dblist.sort(key=int)
