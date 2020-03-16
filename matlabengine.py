import os
from functools import partial

import matlab.engine
import numpy as np

m2np_types = (matlab.double, matlab.single, matlab.int64,
              matlab.int32, matlab.int16, matlab.int8,
              matlab.uint64, matlab.uint32, matlab.uint16,
              matlab.uint8)


def m2np(x):
    if isinstance(x, m2np_types):
        return np.asarray(x._data).reshape(x.size, order='F')
    else:
        return x


class MatRet(object):
    def __init__(self, result, nargout):
        self._ret = result
        self.nargout = nargout

    def __iter__(self):
        if self.nargout > 1:
            for v in self.result:
                yield m2np(v)
        elif self.nargout == 1:
            yield m2np(self.result)

    @property
    def result(self):
        assert isinstance(self._ret, matlab.engine.FutureResult)
        return self._ret.result()


class MatEng(object):
    m_eng = [None for _ in range(2)]
    m_cur = 0
    source_path = os.path.dirname(os.path.abspath(__file__))
    gen_path = [f.path for f in os.scandir(source_path)
                if f.is_dir() and not f.name.startswith('.')]
    start_matlab = partial(matlab.engine.start_matlab, background=True)

    @classmethod
    def start(cls, count=1):
        assert 0 < count <= len(cls.m_eng)
        print(f'launching matlab {count=}')
        for i in range(count):
            if cls.m_eng[i] is None:
                cls.m_eng[i] = cls.start_matlab()

    @classmethod
    def assign_instance(cls):
        _eng = cls.m_eng[cls.m_cur]
        if _eng is None:
            print(f'starting extra matlab {cls.m_cur=}')
            _eng = cls.start_matlab()
        if isinstance(_eng, matlab.engine.FutureResult):
            _eng = _eng.result()
            _eng.addpath(*cls.gen_path)
            cls.m_eng[cls.m_cur] = _eng
        cls.m_cur = (cls.m_cur + 1) % len(cls.m_eng)
        return _eng

    def __init__(self, nargout=1):
        self._eng = self.assign_instance()
        self.nargout = nargout

    def __getattr__(self, item):
        func = getattr(self._eng, item)

        def func_ret(*args, nargout, **kwargs):
            result = func(*args, nargout=nargout, **kwargs)
            return MatRet(result, nargout)

        return partial(func_ret, background=True, nargout=self.nargout)


def Minboundbox(a, nargout=2):
    """
        rotmat, cornerpoints, volume, surface, edgelength
    """
    eng = MatEng(nargout)
    a = (matlab.double(x) for x in np.asarray(a).T)
    return eng.minboundbox(*a, 'v', 3)


def ICP_finite(self, ptarray, pmarray, **options):
    eng = MatEng(2)
    ptarray = matlab.double(np.asarray(ptarray))
    pmarray = matlab.double(np.asarray(pmarray))
    return eng.ICP_finite(ptarray, pmarray, options)
