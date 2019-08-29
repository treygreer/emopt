#include "Grid_cuda.hpp"

#ifndef __CUDA_CLIPPER_HPP__
#define __CUDA_CLIPPER_HPP__

class CudaClipper {
private:
	std::vector<std::complex<double>> _grid;
	std::vector<std::complex<double>> _layer_values;
	std::vector<double> _cell_fractions;
	int _k1, _k2, _j1, _j2, _i1, _i2;
	double _koff, _joff;
	double _dx, _dy;

	void zero_layer_values();
	void compute_ring_cell_fractions(BoostRing ring);
	void composite_cell_fraction(std::complex<double> matval);

public:
	CudaClipper(int k1, int k2, int j1, int j2, int i1, int i2,
				double koff, double joff,
				double dx, double dy);
	void compute_layer_values(StructuredMaterialLayer* layer);
	void composite_layer_values_into_slice(double alpha, int z_index);
	void return_grid_values(std::complex<double> *grid);
};

#endif
