#include "Grid_cuda.hpp"

#ifndef __GRID_CUDA_CTYPES_HPP__
#define __GRID_CUDA_CTYPES_HPP__

using namespace Grid;

// This acts as an interface to the numpy.complex64 data type
typedef struct struct_complex64 {
	double real,
		  imag;
} complex64;

extern "C" {

	////////////////////////////////////////////////////////////////////////////////
	// Polygon
	////////////////////////////////////////////////////////////////////////////////
	Polygon* Polygon_new(double *xs, double *ys, int n,
						 double material_real, double material_imag);
	void Polygon_delete(Polygon* poly);
	//void Polygon_get_points(Polygon* poly, double** x, double** y, int* n);

    ////////////////////////////////////////////////////////////////////////////////
	// Material3D
	////////////////////////////////////////////////////////////////////////////////
	void Material3D_get_value(Material3D* mat, complex64* val, double x, double y, double z);

    void Material3D_get_values(Material3D* mat, complex64* arr, int k1, int k2,
                                                                int j1, int j2, 
                                                                int i1, int i2, 
                                                                double sx, double sy, double sz);

	////////////////////////////////////////////////////////////////////////////////
	// ConstantMaterial3D
	////////////////////////////////////////////////////////////////////////////////
    ConstantMaterial3D* ConstantMaterial3D_new(double real, double imag);

	////////////////////////////////////////////////////////////////////////////////
	// Structured3DMaterial
	////////////////////////////////////////////////////////////////////////////////
    StructuredMaterial3D* StructuredMaterial3D_new(double X, double Y, double Z, double dx, double dy, double dz);
	void StructuredMaterial3D_delete(StructuredMaterial3D* sm);
	void StructuredMaterial3D_add_polygon(StructuredMaterial3D* sm, Polygon* poly, double z1, double z2);
    //void StructuredMaterial3D_get_layers(StructuredMaterial3D* sm, StructuredMaterial2D* layers, double *zs, int *n);

    //void StructuredMaterial2D_get_polys(StructuredMaterial2D* sm2d, Polygon* polys, int *n);
}

#endif