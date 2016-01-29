# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np

cdef extern from "include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int, int) except +
        void set_unary_energy(float*)
        void add_pairwise_energy(float*, float*, int)
        void add_comparison_energy(float*, float*)
        void add_comparison_energy_nystrom(float*, float*, float*)
        void map(int, int*)
        void inference(int, float*)
        int npixels()
        int nlabels()
        int nrank()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int npixels, int nlabels, int nrank):
        self.thisptr = new DenseCRFWrapper(npixels, nlabels, nrank)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:, ::1] unary_costs):
        if (unary_costs.shape[0] != self.thisptr.npixels() or
                unary_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid unary_costs shape")

        self.thisptr.set_unary_energy(&unary_costs[0, 0])

    def add_pairwise_energy(self, float[:, ::1] pairwise_costs,
                            float[:, ::1] features):
        if (pairwise_costs.shape[0] != self.thisptr.nlabels() or
                pairwise_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid pairwise_costs shape")
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")

        self.thisptr.add_pairwise_energy(
            &pairwise_costs[0, 0],
            &features[0, 0],
            features.shape[1]
        )

    def add_comparison_energy(self, float[:, ::1] pair_comp, float[:, ::1] label_comp):
        self.thisptr.add_comparison_energy(&pair_comp[0, 0], &label_comp[0, 0])

    def add_comparison_energy_nystrom(self, float[:, ::1] ns_left, float[:, ::1] ns_right, float[:, ::1] label_comp):
        self.thisptr.add_comparison_energy_nystrom(&ns_left[0, 0], &ns_right[0, 0], &label_comp[0, 0])

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        cdef int[::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0])
        return labels

    def inference(self, int n_iters=10):
        import numpy as np
        Q = np.empty((self.thisptr.npixels(), self.thisptr.nlabels()), dtype=np.float32)
        cdef float[:,::1] Q_view = Q
        self.thisptr.inference(n_iters, &Q_view[0,0])
        return Q
