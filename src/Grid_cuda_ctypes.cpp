#include "Grid_cuda_ctypes.hpp"
#include <string>

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

void Material3D_get_value(Material3D* mat, complex64* val,
						  double fpk, double fpj, double fpi)
{
	int k = std::round(fpk);
	int j = std::round(fpj);
	int i = std::round(fpi);
	double koff = fpk - k;
	double joff = fpj - j;
	double ioff = fpi - i;
	std::vector<std::complex<double>> grid(1);
	std::complex<double> mat_val;
	mat->get_values(grid.data(), k, k+1, j, j+1, i, i+1, koff, joff, ioff);
    val[0].real = grid[0].real();
    val[0].imag = grid[0].imag();
}

void Material3D_get_values(Material3D* mat, complex64* arr,
						   int k1, int k2, 
						   int j1, int j2,
						   int i1, int i2,
						   double koff, double joff, double ioff)
{
    std::complex<double> val;
    int Ny = j2-j1,
        Nx = k2-k1,
        Nz = i2-i1;

	std::vector<std::complex<double>> grid(Nx*Ny*Nz);
    mat->get_values(grid.data(), k1, k2, j1, j2, i1, i2, koff, joff, ioff);

	// convert from std::complex<double> to (Numpy) complex64
    for(int i = 0; i < Nx*Ny*Nz; i++) {
        val = grid[i];
        arr[i].real = val.real();
        arr[i].imag = val.imag();
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
StructuredMaterial3D* StructuredMaterial3D_new(double X, double Y, double Z,
                                               double dx, double dy, double dz)
{
    return new StructuredMaterial3D(X, Y, Z, dx, dy, dz);
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

