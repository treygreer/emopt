#include "Grid_cuda_ctypes.hpp"
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

/////////////////////////////////////////////////////////////////////////////////////
// Polygon Primitives
/////////////////////////////////////////////////////////////////////////////////////

PolyMat* PolyMat_new(double *xs, double *ys, int n,
					 double material_real, double material_imag)
{
	return new PolyMat(xs, ys, n,
					   std::complex<double>(material_real, material_imag));
}

void PolyMat_delete(PolyMat* polymat)
{
	delete polymat;
}

/////////////////////////////////////////////////////////////////////////////////////
// Material3D
/////////////////////////////////////////////////////////////////////////////////////

void Material3D_get_value(Material3D* mat, complex64* val, double x, double y, double z) { 
	std::complex<double> value = mat->get_value(x,y,z);

    val[0].real = std::real(value);
    val[0].imag = std::imag(value);
}

void Material3D_get_values(Material3D* mat, complex64* arr, int k1, int k2, 
                                                            int j1, int j2,
                                                            int i1, int i2,
                                                            double sx, double sy,
                                                            double sz)
{
    std::complex<double> val;
    int Ny = j2-j1,
        Nx = k2-k1,
        Nz = i2-i1;

	ArrayXcd grid(Nx*Ny*Nz);
    mat->get_values(grid, k1, k2, j1, j2, i1, i2, sx, sy, sz);

    for(int i = 0; i < Nx*Ny*Nz; i++) {
        val = grid(i);
        arr[i].real = std::real(val);
        arr[i].imag = std::imag(val);
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// ConstantMaterial3D
/////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial3D* ConstantMaterial3D_new(double real, double imag)
{
    return new ConstantMaterial3D(std::complex<double>(real, imag));
}
void ConstantMaterial3D_delete(ConstantMaterial3D* cm)
{
    delete cm;
}

/////////////////////////////////////////////////////////////////////////////////////
// Structured3DMaterial
/////////////////////////////////////////////////////////////////////////////////////
void StructuredMaterial3D_initialize_class()
{
	StructuredMaterial3D::initialize_class();
}

StructuredMaterial3D* StructuredMaterial3D_new(double X, double Y, double Z,
                                               double dx, double dy, double dz)
{
    return new StructuredMaterial3D(X, Y, Z, dx, dy ,dz);
}

void StructuredMaterial3D_delete(StructuredMaterial3D* sm)
{
    delete sm;
}

void StructuredMaterial3D_add_polymat(StructuredMaterial3D* sm, 
                                      PolyMat* polymat, 
									  double z1, double z2)
{
	sm->add_polymat(polymat, z1, z2);  
}

void StructuredMaterial3D_finalize(StructuredMaterial3D* sm)
{
	sm->finalize();
}
