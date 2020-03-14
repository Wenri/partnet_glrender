import matlab.engine
import numpy as np
import os
from functools import partial


def m2np(x):
    return np.asarray(x._data).reshape(x.size, order='F')


class MatlabEngine(object):
    m_eng = [None for _ in range(2)]
    m_cur = 0
    source_path = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def start(cls, count=1):
        if not count or count < 0 or count >= len(cls.m_eng):
            count = len(cls.m_eng)
        for i in range(count):
            if cls.m_eng[i] is None:
                cls.engine_instance(i)

    @classmethod
    def engine_instance(cls, instance):
        if cls.m_eng[instance] is None:
            print(f'starting new matlab {instance=}')
            cls.m_eng[instance] = matlab.engine.start_matlab(background=True)
        return cls.m_eng[instance]

    @classmethod
    def increment_cur(cls):
        cls.m_cur = (cls.m_cur + 1) % len(cls.m_eng)

    @property
    def eng(self):
        if self._eng is None:
            self._eng = self.engine_instance(self.m_cur)
            self.increment_cur()
        genpath = (f.path for f in os.scandir(self.source_path)
                   if f.is_dir() and not f.name.startswith('.'))
        eng = self._eng.result()
        eng.addpath(*genpath)
        return eng

    def __init__(self, nargout=1):
        self._eng = None
        self._ret = None
        self._nargout = nargout

    def __iter__(self):
        return self

    def __next__(self):
        return

    def __getattr__(self, item):
        func = getattr(self.eng, item)
        return partial(func, background=True, nargout=self._nargout)


class Minboundbox(MatlabEngine):
    def __init__(self, a):
        super().__init__(nargout=2)
        # rotmat, cornerpoints, volume, surface, edgelength
        a = matlab.double(np.asarray(a).T)
        self._ret = self.minboundbox(a[0], a[1], a[2], 'v', 3)

    def __call__(self):
        if self._ret is None:
            return None
        m, corner_points = self._ret.result()
        return np.transpose(m2np(m)), m2np(corner_points)


class ICP_finite(MatlabEngine):
    def __init__(self, ptarray, pmarray, **options):
        super().__init__(nargout=2)
        ptarray = matlab.double(np.asarray(ptarray))
        pmarray = matlab.double(np.asarray(pmarray))
        self.ret = self.ICP_finite(ptarray, pmarray, options)
        self.index = 0

    def __call__(self):
        if self.ret is None:
            return None
        points_moved, m = self.ret.result()
        return m2np(points_moved), m2np(m)
