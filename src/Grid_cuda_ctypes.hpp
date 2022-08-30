#include "Grid_cuda.hpp"

#ifndef __GRID_CUDA_CTYPES_HPP__
#define __GRID_CUDA_CTYPES_HPP__

extern "C" {

	////////////////////////////////////////////////////////////////////////////////
	// Polygon
	////////////////////////////////////////////////////////////////////////////////
	PolyMat* PolyMat_new(double *xs, double *ys, int n,
						 double material);
	void PolyMat_delete(PolyMat* polymat);
	//void Polygon_get_points(Polygon* poly, double** x, double** y, int* n);

    ////////////////////////////////////////////////////////////////////////////////
	// Material3D
	////////////////////////////////////////////////////////////////////////////////
	void Material3D_get_value(Material3D* mat, double* val, double x, double y, double z);

    void Material3D_get_values(Material3D* mat, double* arr, int k1, int k2,
                                                                int j1, int j2, 
                                                                int i1, int i2, 
                                                                double koff, double joff, double ioff);

	////////////////////////////////////////////////////////////////////////////////
	// ConstantMaterial3D
	////////////////////////////////////////////////////////////////////////////////
    ConstantMaterial3D* ConstantMaterial3D_new(double val);
	void ConstantMaterial3D_delete(ConstantMaterial3D* cm);

	////////////////////////////////////////////////////////////////////////////////
	// Structured3DMaterial
	////////////////////////////////////////////////////////////////////////////////
    StructuredMaterial3D* StructuredMaterial3D_new(double X, double Y, double Z, double dx, double dy, double dz);
	void StructuredMaterial3D_delete(StructuredMaterial3D* sm);
	void StructuredMaterial3D_add_polymat(StructuredMaterial3D* sm, PolyMat* polymat, double z1, double z2);
    //void StructuredMaterial3D_get_layers(StructuredMaterial3D* sm, StructuredMaterial2D* layers, double *zs, int *n);

    //void StructuredMaterial2D_get_polys(StructuredMaterial2D* sm2d, Polygon* polys, int *n);
}

#endif
