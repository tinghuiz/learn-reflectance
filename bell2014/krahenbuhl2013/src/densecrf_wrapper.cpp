#include <Eigen/Core>
#include "densecrf.h"
#include "pairwise.h"
#include "densecrf_wrapper.h"

DenseCRFWrapper::DenseCRFWrapper(int npixels, int nlabels, int nrank)
: m_npixels(npixels), m_nlabels(nlabels), m_nrank(nrank) {
	m_crf = new DenseCRF(npixels, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	delete m_crf;
}

int DenseCRFWrapper::npixels() { return m_npixels; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }

void DenseCRFWrapper::add_pairwise_energy(float* pairwise_costs_ptr, float* features_ptr, int nfeatures) {
	m_crf->addPairwiseEnergy(
		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
		new MatrixCompatibility(
			Eigen::Map<const Eigen::MatrixXf>(pairwise_costs_ptr, m_nlabels, m_nlabels)
		),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC
	);
}

void DenseCRFWrapper::add_comparison_energy(float* pair_comp_ptr, float* label_comp_ptr) {
	
	PairwisePotential* comp_potential_ptr = new PairwisePotential(new MatrixCompatibility( 
		Eigen::Map<const Eigen::MatrixXf>(label_comp_ptr, m_nlabels, m_nlabels) ), 
		Eigen::Map<const Eigen::MatrixXf>(pair_comp_ptr, m_npixels, m_npixels));
	m_crf->addPairwiseEnergy(comp_potential_ptr);
}

void DenseCRFWrapper::add_comparison_energy_nystrom(float* ns_left_ptr, float* ns_right_ptr, float* label_comp_ptr) {
	
	PairwisePotential* comp_potential_ns_ptr = new PairwisePotential(new MatrixCompatibility( 
		Eigen::Map<const Eigen::MatrixXf>(label_comp_ptr, m_nlabels, m_nlabels) ), 
		Eigen::Map<const Eigen::MatrixXf>(ns_left_ptr, m_npixels, m_nrank), 
		Eigen::Map<const Eigen::MatrixXf>(ns_right_ptr, m_nrank, m_npixels) );
	m_crf->addPairwiseEnergy(comp_potential_ns_ptr);
}

void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, m_npixels)
	);
}

void DenseCRFWrapper::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < m_npixels; i ++)
		labels[i] = labels_vec(i);
}

void DenseCRFWrapper::inference(int n_iters, float* Q) {
	MatrixXf Q_vec = m_crf->inference(n_iters);
	memcpy(Q, Q_vec.data(), Q_vec.cols()*Q_vec.rows()*sizeof(float));
}
