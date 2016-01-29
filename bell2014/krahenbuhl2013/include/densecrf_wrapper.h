#include "densecrf.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int npixels, int nlabels, int nrank);
		virtual ~DenseCRFWrapper();

		void set_unary_energy(float* unary_costs_ptr);

		void add_pairwise_energy(float* pairwise_costs_ptr,
				float* features_ptr, int nfeatures);

		void add_comparison_energy(float* pair_comp_ptr, float* label_comp_ptr);

		void add_comparison_energy_nystrom(float* ns_left_ptr, float* ns_right_ptr, float* label_comp_ptr);

		void map(int n_iters, int* result);

		void inference(int n_iters, float* Q);

		int npixels();
		int nlabels();
		int nrank();

	private:
		DenseCRF* m_crf;
		int m_npixels;
		int m_nlabels;
		int m_nrank;
};
