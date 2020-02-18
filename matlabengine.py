import matlab.engine
import numpy as np


class MatlabEngine(object):
    m_eng = [None for _ in range(2)]
    m_cur = 0

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
        return self._eng.result()

    def __init__(self):
        self._eng = None


def mat_to_ndarray(x):
    return np.asarray(x._data).reshape(x.size, order='F')


class Minboundbox(MatlabEngine):
    def __init__(self):
        super().__init__()
        self.ret = None

    def __call__(self, a):
        # rotmat, cornerpoints, volume, surface, edgelength
        a = matlab.double(np.asarray(a).T)
        self.ret = self.eng.minboundbox(a[0], a[1], a[2], 'v', 1,
                                        background=True, nargout=1)
        return self

    def result(self):
        if self.ret is None:
            return None
        return mat_to_ndarray(self.ret.result())


class ICP_finite(MatlabEngine):
    def __init__(self):
        super().__init__()
        self.ret = None

    def __call__(self, ptarray, pmarray, **kwargs):
        ptarray = matlab.double(np.asarray(ptarray))
        pmarray = matlab.double(np.asarray(pmarray))
        self.ret = self.eng.ICP_finite(ptarray, pmarray, kwargs,
                                       background=True, nargout=2)
        return self

    def result(self):
        if self.ret is None:
            return None
        points_moved, m = self.ret.result()
        return mat_to_ndarray(points_moved), mat_to_ndarray(m)
