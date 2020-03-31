# cython: language_level=3

from libc.stdint cimport uintptr_t
import pcl
cimport pcl

cdef extern from "pcsimilarity.hpp":
    float similarity[CloudT](const CloudT* cloudA, const CloudT* cloudB)

cdef float _pclsimilarity(pcl.PointCloud cloudA, pcl.PointCloud cloudB):
    return similarity(cloudA.thisptr(), cloudB.thisptr())

def pclsimilarity(cloudA: pcl.PointCloud, cloudB: pcl.PointCloud) -> float:
    return _pclsimilarity(cloudA, cloudB)
