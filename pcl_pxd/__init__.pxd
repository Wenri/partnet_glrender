cimport pcl.pcl_defs as cpp

# class override(PointCloud)
cdef class PointCloud:
    cdef cpp.PointCloudPtr_t thisptr_shared  # XYZ

    # Buffer protocol support.
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _view_count

    cdef inline cpp.PointCloud[cpp.PointXYZ] *thisptr(self) nogil:
        # Shortcut to get raw pointer to underlying PointCloud<PointXYZ>.
        return self.thisptr_shared.get()
