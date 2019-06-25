#include "fdtd_cuda.hpp"
#include <math.h>
#include <algorithm>
#undef NDEBUG
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

fdtd::FDTD::FDTD(int Nx, int Ny, int Nz)
{
    _Nx = Nx;
    _Ny = Ny;
    _Nz = Nz;

	// Allocate field arrays
	int N = Nz * Ny * Nx;
	cudaMallocManaged((void **)&_Ex, N*sizeof(double));
	cudaMallocManaged((void **)&_Ey, N*sizeof(double));
	cudaMallocManaged((void **)&_Ez, N*sizeof(double));
	cudaMallocManaged((void **)&_Hx, N*sizeof(double));
	cudaMallocManaged((void **)&_Hy, N*sizeof(double));
	cudaMallocManaged((void **)&_Hz, N*sizeof(double));

	// Allocate material arrays
	cudaMallocManaged((void **)&_eps_x, N*sizeof(complex128));
	cudaMallocManaged((void **)&_eps_y, N*sizeof(complex128));
	cudaMallocManaged((void **)&_eps_z, N*sizeof(complex128));
	cudaMallocManaged((void **)&_mu_x, N*sizeof(complex128));
	cudaMallocManaged((void **)&_mu_y, N*sizeof(complex128));
	cudaMallocManaged((void **)&_mu_z, N*sizeof(complex128));

	std::cout << "after malloc, _mu_z: " << _mu_z << std::endl;	

    // make sure all of our PML arrays start NULL
    _pml_Exy0 = NULL; _pml_Exy1 = NULL; _pml_Exz0 = NULL; _pml_Exz1 = NULL;
    _pml_Eyx0 = NULL; _pml_Eyx1 = NULL; _pml_Eyz0 = NULL; _pml_Eyz1 = NULL;
    _pml_Ezx0 = NULL; _pml_Ezx1 = NULL; _pml_Ezy0 = NULL; _pml_Ezy1 = NULL;
    _pml_Hxy0 = NULL; _pml_Hxy1 = NULL; _pml_Hxz0 = NULL; _pml_Hxz1 = NULL;
    _pml_Hyx0 = NULL; _pml_Hyx1 = NULL; _pml_Hyz0 = NULL; _pml_Hyz1 = NULL;
    _pml_Hzx0 = NULL; _pml_Hzx1 = NULL; _pml_Hzy0 = NULL; _pml_Hzy1 = NULL;

    _kappa_H_x = NULL; _kappa_H_y = NULL; _kappa_H_z = NULL;
    _kappa_E_x = NULL; _kappa_E_y = NULL; _kappa_E_z = NULL;

    _bHx = NULL; _bHy = NULL; _bHz = NULL;
    _bEx = NULL; _bEy = NULL; _bEz = NULL;

    _cHx = NULL; _cHy = NULL; _cHz = NULL;
    _cEx = NULL; _cEy = NULL; _cEz = NULL;

    _w_pml_x0 = 0; _w_pml_x1 = 0;
    _w_pml_y0 = 0; _w_pml_y1 = 0;
    _w_pml_z0 = 0; _w_pml_z1 = 0;

    _complex_eps = false;
}

fdtd::FDTD::~FDTD()
{
	// Clean up Field arrays
	cudaFree(_Ex); cudaFree(_Ey); cudaFree(_Ez);
	cudaFree(_Hx); cudaFree(_Hy); cudaFree(_Hz);

	// Clean up Material arrays
	cudaFree(_eps_x); cudaFree(_eps_y); cudaFree(_eps_z);
	cudaFree(_mu_x); cudaFree(_mu_y); cudaFree(_mu_z);

    // Clean up PML arrays
    cudaFree(_pml_Exy0); cudaFree(_pml_Exy1); cudaFree(_pml_Exz0); cudaFree(_pml_Exz1);
    cudaFree(_pml_Eyx0); cudaFree(_pml_Eyx1); cudaFree(_pml_Eyz0); cudaFree(_pml_Eyz1);
    cudaFree(_pml_Ezx0); cudaFree(_pml_Ezx1); cudaFree(_pml_Ezy0); cudaFree(_pml_Ezy1);
    cudaFree(_pml_Hxy0); cudaFree(_pml_Hxy1); cudaFree(_pml_Hxz0); cudaFree(_pml_Hxz1);
    cudaFree(_pml_Hyx0); cudaFree(_pml_Hyx1); cudaFree(_pml_Hyz0); cudaFree(_pml_Hyz1);
    cudaFree(_pml_Hzx0); cudaFree(_pml_Hzx1); cudaFree(_pml_Hzy0); cudaFree(_pml_Hzy1);

    cudaFree(_kappa_H_x);
    cudaFree(_kappa_H_y);
    cudaFree(_kappa_H_z);

    cudaFree(_kappa_E_x);
    cudaFree(_kappa_E_y);
    cudaFree(_kappa_E_z);

    cudaFree(_bHx);
    cudaFree(_bHy);
    cudaFree(_bHz);

    cudaFree(_bEx);
    cudaFree(_bEy);
    cudaFree(_bEz);

    cudaFree(_cHx);
    cudaFree(_cHy);
    cudaFree(_cHz);

    cudaFree(_cEx);
    cudaFree(_cEy);
    cudaFree(_cEz);
}

void fdtd::FDTD::set_physical_dims(double X, double Y, double Z,
								   double dx, double dy, double dz)
{
    _X = X; _Y = Y; _Z = Z;
    _dx = dx; _dy = dy; _dz = dz;
}

void fdtd::FDTD::set_wavelength(double wavelength)
{
    _wavelength = wavelength;
    _R = _wavelength/(2*M_PI);
}


void fdtd::FDTD::set_dt(double dt)
{
    _dt = dt;
    _odt = 1.0/_dt;
}

void fdtd::FDTD::set_complex_eps(bool complex_eps)
{
    _complex_eps = complex_eps;
}

__device__
double cuda_src_func_t(double t, double phase, double src_T, double src_min, double src_k)
{
    if(t <= src_T)
        return sin(t + phase)*((1+src_min) * exp(-(t-src_T)*(t-src_T) / src_k) - src_min);
    else
        return sin(t + phase);
}

__global__
void update_H_fields(double t, double R,
					 int Nx, int Ny, int Nz,
					 double dx, double dy, double dz, double dt,
					 char bc0, char bc1, char bc2,
					 int w_pml_x0, int w_pml_x1,
					 int w_pml_y0, int w_pml_y1,
					 int w_pml_z0, int w_pml_z1,
					 double *Ex, double *Ey, double *Ez,
					 double *Hx, double *Hy, double *Hz,
					 complex128 *mu_x, complex128 *mu_y, complex128 *mu_z,
					 double *pml_Exy0, double *pml_Exy1, double *pml_Exz0, double *pml_Exz1,
					 double *pml_Eyx0, double *pml_Eyx1, double *pml_Eyz0, double *pml_Eyz1,
					 double *pml_Ezx0, double *pml_Ezx1, double *pml_Ezy0, double *pml_Ezy1,
					 double *kappa_H_x, double *kappa_H_y, double *kappa_H_z,
					 double *bHx, double *bHy, double *bHz,
					 double *cHx, double *cHy, double *cHz
	)
{
    double odx = R/dx,
           ody = R/dy,
           odz = R/dz,
           b, C, kappa,
           dt_by_mux, dt_by_muy, dt_by_muz;

    int pml_xmin = w_pml_x0, pml_xmax = Nx-w_pml_x1,
        pml_ymin = w_pml_y0, pml_ymax = Ny-w_pml_y1,
        pml_zmin = w_pml_z0, pml_zmax = Nz-w_pml_z1;

    int ind_ijk, ind_ip1jk, ind_ijp1k, ind_ijkp1,
        ind_pml, 
        ind_pml_param;

    double dExdy, dExdz, dEydx, dEydz, dEzdx, dEzdy;

	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < Nz) && (j < Ny) && (k < Nx)) {

        int kill_zwrap = (bc2 != 'P' && i == Nz-1) ? 1 : 0;
		int ip1 = i==Nz-1 ? 0 : i+1;

		int kill_ywrap = (bc1 != 'P' && j == Ny-1) ? 1 : 0;
		int jp1 = j==Ny-1 ? 0 : j+1;

		int kill_xwrap = (bc0 != 'P' && k == Nx-1) ? 1 : 0;
		int kp1 = k==Nx-1 ? 0 : k+1;

		ind_ijk =   (i+0)*Ny*Nx + (j+0)*Nx + (k+0);
		ind_ijp1k = (i+0)*Ny*Nx + (jp1)*Nx + (k+0);
		ind_ip1jk = (ip1)*Ny*Nx + (j+0)*Nx + (k+0);
		ind_ijkp1 = (i+0)*Ny*Nx + (j+0)*Nx + (kp1);

		// compute prefactors
		dt_by_mux = dt/mu_x[ind_ijk].real;
		dt_by_muy = dt/mu_y[ind_ijk].real;
		dt_by_muz = dt/mu_z[ind_ijk].real;

		// Update Hx
		dEzdy = ody * ((kill_ywrap ? 0 : Ez[ind_ijp1k])  - Ez[ind_ijk]);
		dEydz = odz * ((kill_zwrap ? 0 : Ey[ind_ip1jk])  - Ey[ind_ijk]);
		Hx[ind_ijk] += dt_by_mux * (dEydz - dEzdy);

		// update Hy
		dExdz = odz * ((kill_zwrap ? 0 : Ex[ind_ip1jk]) - Ex[ind_ijk]);
		dEzdx = odx * ((kill_xwrap ? 0 : Ez[ind_ijkp1]) - Ez[ind_ijk]);
		Hy[ind_ijk] += dt_by_muy * (dEzdx - dExdz);

		// update Hz
		dEydx = odx * ((kill_xwrap ? 0 : Ey[ind_ijkp1]) - Ey[ind_ijk]);
		dExdy = ody * ((kill_ywrap ? 0 : Ex[ind_ijp1k]) - Ex[ind_ijk]);
		Hz[ind_ijk] += dt_by_muz * (dExdy - dEydx);

		// Do PML updates
		if(k < pml_xmin) {
			// get index in PML array
			ind_pml = i*Ny*(pml_xmin) +j*(pml_xmin) + k;

			// get PML coefficients
			ind_pml_param = pml_xmin - k - 1;
			kappa = kappa_H_x[ind_pml_param];
			b = bHx[ind_pml_param];
			C = cHx[ind_pml_param];

			// Update PML convolution
			pml_Eyx0[ind_pml] = C * dEydx + b*pml_Eyx0[ind_pml];
			pml_Ezx0[ind_pml] = C * dEzdx + b*pml_Ezx0[ind_pml];

			Hz[ind_ijk] -= dt_by_muz * (pml_Eyx0[ind_pml]-dEydx+dEydx/kappa);
			Hy[ind_ijk] += dt_by_muy * (pml_Ezx0[ind_pml]-dEzdx+dEzdx/kappa);

		}
		else if(k  >= pml_xmax) {
			ind_pml = i*Ny*(Nx - pml_xmax) + j*(Nx - pml_xmax) + k - pml_xmax;

			// get pml coefficients
			ind_pml_param = k - pml_xmax + w_pml_x0;
			kappa = kappa_H_x[ind_pml_param];
			b = bHx[ind_pml_param];
			C = cHx[ind_pml_param];

			pml_Eyx1[ind_pml] = C * dEydx + b*pml_Eyx1[ind_pml];
			pml_Ezx1[ind_pml] = C * dEzdx + b*pml_Ezx1[ind_pml];

			Hz[ind_ijk] -= dt_by_muz * (pml_Eyx1[ind_pml]-dEydx+dEydx/kappa);
			Hy[ind_ijk] += dt_by_muy * (pml_Ezx1[ind_pml]-dEzdx+dEzdx/kappa);
		}

		if(j < pml_ymin) {
			ind_pml = i*pml_ymin*Nx +j*Nx + k;

			// compute coefficients
			ind_pml_param = pml_ymin - j - 1;
			kappa = kappa_H_y[ind_pml_param];
			b = bHy[ind_pml_param];
			C = cHy[ind_pml_param];

			pml_Exy0[ind_pml] = C * dExdy + b*pml_Exy0[ind_pml];
			pml_Ezy0[ind_pml] = C * dEzdy + b*pml_Ezy0[ind_pml];

			Hz[ind_ijk] += dt_by_muz * (pml_Exy0[ind_pml]-dExdy+dExdy/kappa);
			Hx[ind_ijk] -= dt_by_mux * (pml_Ezy0[ind_pml]-dEzdy+dEzdy/kappa);
		}
		else if(j >= pml_ymax) {
			ind_pml = i*(Ny - pml_ymax)*Nx +(j - pml_ymax)*Nx + k;

			// compute coefficients
			ind_pml_param = j - pml_ymax + w_pml_y0;
			kappa = kappa_H_y[ind_pml_param];
			b = bHy[ind_pml_param];
			C = cHy[ind_pml_param];

			pml_Exy1[ind_pml] = C * dExdy + b*pml_Exy1[ind_pml];
			pml_Ezy1[ind_pml] = C * dEzdy + b*pml_Ezy1[ind_pml];

			Hz[ind_ijk] += dt_by_muz * (pml_Exy1[ind_pml]-dExdy+dExdy/kappa);
			Hx[ind_ijk] -= dt_by_mux * (pml_Ezy1[ind_pml]-dEzdy+dEzdy/kappa);
		}

		if(i < pml_zmin) {
			ind_pml = i*Ny*Nx +j*Nx + k;

			// get coefficients
			ind_pml_param = pml_zmin - i - 1;
			kappa = kappa_H_z[ind_pml_param];
			b = bHz[ind_pml_param];
			C = cHz[ind_pml_param];

			pml_Exz0[ind_pml] = C * dExdz + b*pml_Exz0[ind_pml];
			pml_Eyz0[ind_pml] = C * dEydz + b*pml_Eyz0[ind_pml];

			Hx[ind_ijk] += dt_by_mux * (pml_Eyz0[ind_pml]-dEydz+dEydz/kappa);
			Hy[ind_ijk] -= dt_by_muy * (pml_Exz0[ind_pml]-dExdz+dExdz/kappa);
		}
		else if(i > pml_zmax) {
			ind_pml = (i - pml_zmax)*Ny*Nx +j*Nx + k;

			// get coefficients
			ind_pml_param = i - pml_zmax + w_pml_z0;
			kappa = kappa_H_z[ind_pml_param];
			b = bHz[ind_pml_param];
			C = cHz[ind_pml_param];

			pml_Exz1[ind_pml] = C * dExdz + b*pml_Exz1[ind_pml];
			pml_Eyz1[ind_pml] = C * dEydz + b*pml_Eyz1[ind_pml];

			Hx[ind_ijk] += dt_by_mux * (pml_Eyz1[ind_pml]-dEydz+dEydz/kappa);
			Hy[ind_ijk] -= dt_by_muy * (pml_Exz1[ind_pml]-dExdz+dExdz/kappa);
		}

	}
}

__global__
void update_H_sources(double t,
					  int Nx, int Ny, int Nz,
					  double dt,
					  double *Hx, double *Hy, double *Hz,
					  complex128 *mu_x, complex128 *mu_y, complex128 *mu_z,
					  double src_T, double src_min, double src_k,
					  int i0s, int j0s, int k0s,
					  int Is, int Js, int Ks,
					  complex128 *Mx, complex128 *My, complex128 *Mz)
{
    int ind_ijk, ind_src;
    double src_t;

	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < Is) && (j < Js) && (k < Ks)) {
		ind_ijk = (i+i0s)*Ny*Nx + (j+j0s)*Nx + (k+k0s);
		ind_src = i*Js*Ks + j*Ks + k;

        // update Mx
		src_t = cuda_src_func_t(t, Mx[ind_src].imag,
								src_T, src_min, src_k);
		Hx[ind_ijk] += src_t * Mx[ind_src].real * dt / mu_x[ind_ijk].real;

        // update My
		src_t = cuda_src_func_t(t, My[ind_src].imag,
								src_T, src_min, src_k);
		Hy[ind_ijk] += src_t * My[ind_src].real * dt / mu_y[ind_ijk].real;

        // update Mz
		src_t = cuda_src_func_t(t, Mz[ind_src].imag,
								src_T, src_min, src_k);
		Hz[ind_ijk] += src_t * Mz[ind_src].real * dt / mu_z[ind_ijk].real;
    }
}

void fdtd::FDTD::update_H(double t)
{
	dim3 fields_threadsPerBlock(8, 8, 8);
	dim3 fields_numBlocks(ceil((float)_Nx/fields_threadsPerBlock.x),
						  ceil((float)_Ny/fields_threadsPerBlock.y),
						  ceil((float)_Nz/fields_threadsPerBlock.z));

	update_H_fields <<<fields_numBlocks, fields_threadsPerBlock>>>
		(t, _R,
		 _Nx, _Ny, _Nz,
		 _dx, _dy, _dz, _dt,
		 _bc[0], _bc[1], _bc[2],
		 _w_pml_x0, _w_pml_x1,
		 _w_pml_y0, _w_pml_y1,
		 _w_pml_z0, _w_pml_z1,
		 _Ex, _Ey, _Ez,
		 _Hx, _Hy, _Hz,
		 _mu_x, _mu_y, _mu_z,
		 _pml_Exy0, _pml_Exy1, _pml_Exz0, _pml_Exz1,
		 _pml_Eyx0, _pml_Eyx1, _pml_Eyz0, _pml_Eyz1,
		 _pml_Ezx0, _pml_Ezx1, _pml_Ezy0, _pml_Ezy1,
		 _kappa_H_x, _kappa_H_y, _kappa_H_z,
		 _bHx, _bHy, _bHz,
		 _cHx, _cHy, _cHz);

    // Update sources
    for(auto const& src : _sources) {
		dim3 sources_threadsPerBlock(8, 8, 8);
		dim3 sources_numBlocks(ceil((float) src.K/sources_threadsPerBlock.x),
							   ceil((float) src.J/sources_threadsPerBlock.y),
							   ceil((float) src.I/sources_threadsPerBlock.z));

		update_H_sources <<<sources_numBlocks, sources_threadsPerBlock>>>
			(t,
			 _Nx, _Ny, _Nz,
			 _dt,
			 _Hx, _Hy, _Hz,
			 _mu_x, _mu_y, _mu_z,
			 _src_T, _src_min, _src_k,
			 src.i0, src.j0, src.k0,
			 src.I, src.J, src.K,
			 src.Mx, src.My, src.Mz);
	}
}

enum action { ACTION_NOP, ACTION_ZERO, ACTION_FLIP, ACTION_COPY };

__global__
void update_E_fields(double t, double R,
					 int Nx, int Ny, int Nz,
					 double dx, double dy, double dz, double dt,
					 char bc0, char bc1, char bc2,
					 int w_pml_x0, int w_pml_x1,
					 int w_pml_y0, int w_pml_y1,
					 int w_pml_z0, int w_pml_z1,
					 double *Ex, double *Ey, double *Ez,
					 double *Hx, double *Hy, double *Hz,
					 complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
					 double *pml_Hxy0, double *pml_Hxy1, double *pml_Hxz0, double *pml_Hxz1,
					 double *pml_Hyx0, double *pml_Hyx1, double *pml_Hyz0, double *pml_Hyz1,
					 double *pml_Hzx0, double *pml_Hzx1, double *pml_Hzy0, double *pml_Hzy1,
					 double *kappa_E_x, double *kappa_E_y, double *kappa_E_z,
					 double *bEx, double *bEy, double *bEz,
					 double *cEx, double *cEy, double *cEz
	)
{
    double odx = R/dx,
           ody = R/dy,
		   odz = R/dz,
		b_x, b_y, b_z;

    int ind_ijk, ind_im1jk, ind_ijm1k, ind_ijkm1, ind_pml, ind_pml_param;

    double dHxdy, dHxdz, dHydx, dHydz, dHzdx, dHzdy;

    double b, C, kappa;

    int pml_xmin = w_pml_x0, pml_xmax = Nx-w_pml_x1,
        pml_ymin = w_pml_y0, pml_ymax = Ny-w_pml_y1,
        pml_zmin = w_pml_z0, pml_zmax = Nz-w_pml_z1;

	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < Nz) && (j < Ny) && (k < Nx)) {
        int action_zwrap = (bc2 == 'P' || i != 0 ? ACTION_NOP :
							bc2 == '0'           ? ACTION_ZERO :
							bc2 == 'E'           ? ACTION_FLIP : ACTION_COPY);
		int im1 = i==0 ? Nz-1 : i-1;
		int action_ywrap = (bc1 == 'P' || j != 0 ? ACTION_NOP :
							bc1 == '0'           ? ACTION_ZERO :
							bc1 == 'E'           ? ACTION_FLIP : ACTION_COPY);
		int jm1 = j==0 ? Ny-1 : j-1;
		int action_xwrap = (bc0 == 'P' || k != 0 ? ACTION_NOP :
							bc0 == '0'           ? ACTION_ZERO :
							bc0 == 'E'           ? ACTION_FLIP : ACTION_COPY);
		int km1 = k==0 ? Nx-1 : k-1;

		ind_ijk   = (i-0)*Ny*Nx + (j-0)*Nx + (k-0);
		ind_ijm1k = (i-0)*Ny*Nx + (jm1)*Nx + (k-0);
		ind_im1jk = (im1)*Ny*Nx + (j-0)*Nx + (k-0);
		ind_ijkm1 = (i-0)*Ny*Nx + (j-0)*Nx + (km1);

		b_x = dt/eps_x[ind_ijk].real;
		b_y = dt/eps_y[ind_ijk].real;
		b_z = dt/eps_z[ind_ijk].real;

		// Update Ex
		dHzdy = ody*(Hz[ind_ijk] - (action_ywrap == ACTION_ZERO ? 0.0 :
									action_ywrap == ACTION_COPY ?  Hz[ind_ijk] :
									action_ywrap == ACTION_FLIP ? -Hz[ind_ijk] : Hz[ind_ijm1k]));
		dHydz = odz*(Hy[ind_ijk] - (action_zwrap == ACTION_ZERO ? 0.0 :
									action_zwrap == ACTION_COPY ?  Hy[ind_ijk] :
									action_zwrap == ACTION_FLIP ? -Hy[ind_ijk] : Hy[ind_im1jk]));
		Ex[ind_ijk] = Ex[ind_ijk] + (dHzdy - dHydz) * b_x;

		// Update Ey
		dHxdz = odz*(Hx[ind_ijk] - (action_zwrap == ACTION_ZERO ? 0.0 :
									action_zwrap == ACTION_COPY ?  Hx[ind_ijk] :
									action_zwrap == ACTION_FLIP ? -Hx[ind_ijk] : Hx[ind_im1jk]));
		dHzdx = odx*(Hz[ind_ijk] - (action_xwrap == ACTION_ZERO ? 0.0 :
									action_xwrap == ACTION_COPY ?  Hz[ind_ijk] :
									action_xwrap == ACTION_FLIP ? -Hz[ind_ijk] : Hz[ind_ijkm1]));
		Ey[ind_ijk] = Ey[ind_ijk] + (dHxdz - dHzdx) * b_y;

		// Update Ez
		dHydx = odx*(Hy[ind_ijk] - (action_xwrap == ACTION_ZERO ? 0.0 :
									action_xwrap == ACTION_COPY ?  Hy[ind_ijk] :
									action_xwrap == ACTION_FLIP ? -Hy[ind_ijk] : Hy[ind_ijkm1]));
		dHxdy = ody*(Hx[ind_ijk] - (action_ywrap == ACTION_ZERO ? 0.0 :
									action_ywrap == ACTION_COPY ?  Hx[ind_ijk] :
									action_ywrap == ACTION_FLIP ? -Hx[ind_ijk] : Hx[ind_ijm1k]));
		Ez[ind_ijk] = Ez[ind_ijk] + (dHydx - dHxdy) * b_z;

		// Do PML updates
		if(k < pml_xmin) {
			ind_pml = i*Ny*(pml_xmin) +j*(pml_xmin) + k;

			// get PML coefficients
			ind_pml_param = pml_xmin - k - 1;
			kappa = kappa_E_x[ind_pml_param];
			b = bEx[ind_pml_param];
			C = cEx[ind_pml_param];

			pml_Hyx0[ind_pml] = C * dHydx + b*pml_Hyx0[ind_pml];
			pml_Hzx0[ind_pml] = C * dHzdx + b*pml_Hzx0[ind_pml];

			Ez[ind_ijk] = Ez[ind_ijk] + (pml_Hyx0[ind_pml]-dHydx+dHydx/kappa) * b_z;
			Ey[ind_ijk] = Ey[ind_ijk] - (pml_Hzx0[ind_pml]-dHzdx+dHzdx/kappa) * b_y;

		}
		else if(k >= pml_xmax) {
			ind_pml = i*Ny*(Nx - pml_xmax) +j*(Nx - pml_xmax) + k - pml_xmax;

			// get coefficients
			ind_pml_param = k - pml_xmax + w_pml_x0;
			kappa = kappa_E_x[ind_pml_param];
			b = bEx[ind_pml_param];
			C = cEx[ind_pml_param];

			pml_Hyx1[ind_pml] = C * dHydx + b*pml_Hyx1[ind_pml];
			pml_Hzx1[ind_pml] = C * dHzdx + b*pml_Hzx1[ind_pml];

			Ez[ind_ijk] = Ez[ind_ijk] + (pml_Hyx1[ind_pml]-dHydx+dHydx/kappa) * b_z;
			Ey[ind_ijk] = Ey[ind_ijk] - (pml_Hzx1[ind_pml]-dHzdx+dHzdx/kappa) * b_y;
		}

		if(j < pml_ymin) {
			ind_pml = i*pml_ymin*Nx +j*Nx + k;

			// get coefficients
			ind_pml_param = pml_ymin - j - 1;
			kappa = kappa_E_y[ind_pml_param];
			b = bEy[ind_pml_param];
			C = cEy[ind_pml_param];

			pml_Hxy0[ind_pml] = C * dHxdy + b*pml_Hxy0[ind_pml];
			pml_Hzy0[ind_pml] = C * dHzdy + b*pml_Hzy0[ind_pml];

			Ez[ind_ijk] = Ez[ind_ijk] - (pml_Hxy0[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
			Ex[ind_ijk] = Ex[ind_ijk] + (pml_Hzy0[ind_pml]-dHzdy+dHzdy/kappa) * b_x;
		}
		else if(j >= pml_ymax) {
			ind_pml = i*(Ny - pml_ymax)*Nx +(j - pml_ymax)*Nx + k;

			// get coefficients
			ind_pml_param = j - pml_ymax + w_pml_y0;
			kappa = kappa_E_y[ind_pml_param];
			b = bEy[ind_pml_param];
			C = cEy[ind_pml_param];

			pml_Hxy1[ind_pml] = C * dHxdy + b*pml_Hxy1[ind_pml];
			pml_Hzy1[ind_pml] = C * dHzdy + b*pml_Hzy1[ind_pml];

			Ez[ind_ijk] = Ez[ind_ijk] - (pml_Hxy1[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
			Ex[ind_ijk] = Ex[ind_ijk] + (pml_Hzy1[ind_pml]-dHzdy+dHzdy/kappa) * b_x;
		}

		if(i < pml_zmin) {
			ind_pml = i*Ny*Nx +j*Nx + k;

			// get coefficients
			ind_pml_param = pml_zmin - i - 1;
			kappa = kappa_E_z[ind_pml_param];
			b = bEz[ind_pml_param];
			C = cEz[ind_pml_param];

			pml_Hxz0[ind_pml] = C * dHxdz + b*pml_Hxz0[ind_pml];
			pml_Hyz0[ind_pml] = C * dHydz + b*pml_Hyz0[ind_pml];

			Ex[ind_ijk] = Ex[ind_ijk] - (pml_Hyz0[ind_pml]-dHydz+dHydz/kappa) * b_x;
			Ey[ind_ijk] = Ey[ind_ijk] + (pml_Hxz0[ind_pml]-dHxdz+dHxdz/kappa) * b_y;
		}
		else if(i > pml_zmax) {
			ind_pml = (i - pml_zmax)*Ny*Nx +j*Nx + k;

			// compute coefficients
			ind_pml_param = i - pml_zmax + w_pml_z0;
			kappa = kappa_E_z[ind_pml_param];
			b = bEz[ind_pml_param];
			C = cEz[ind_pml_param];

			pml_Hxz1[ind_pml] = C * dHxdz + b*pml_Hxz1[ind_pml];
			pml_Hyz1[ind_pml] = C * dHydz + b*pml_Hyz1[ind_pml];

			Ex[ind_ijk] = Ex[ind_ijk] - (pml_Hyz1[ind_pml]-dHydz+dHydz/kappa) * b_x;
			Ey[ind_ijk] = Ey[ind_ijk] + (pml_Hxz1[ind_pml]-dHxdz+dHxdz/kappa) * b_y;
		}
	}
}

__global__
void update_E_sources(double t,
					  int Nx, int Ny, int Nz,
					  double dt,
					  double *Ex, double *Ey, double *Ez,
					  complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
					  double src_T, double src_min, double src_k,
					  int i0s, int j0s, int k0s,
					  int Is, int Js, int Ks,
					  complex128 *Jx, complex128 *Jy, complex128 *Jz)
{
    int ind_ijk, ind_src;
    double src_t;
	double b;

	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < Is) && (j < Js) && (k < Ks)) {

		ind_ijk = (i+i0s)*Ny*Nx + (j+j0s)*Nx + (k+k0s);
		ind_src = i*Js*Ks + j*Ks + k;

		// update Jx
		b = dt/eps_x[ind_ijk].real;
		src_t = cuda_src_func_t(t, Jx[ind_src].imag,
								src_T, src_min, src_k);
		Ex[ind_ijk] -= src_t * Jx[ind_src].real * b;

		// update Jy
		b = dt/eps_y[ind_ijk].real;
		src_t = cuda_src_func_t(t, Jy[ind_src].imag,
								src_T, src_min, src_k);
		Ey[ind_ijk] -= src_t * Jy[ind_src].real * b;

		// update Jz
		b = dt/eps_z[ind_ijk].real;
		src_t = cuda_src_func_t(t, Jz[ind_src].imag,
								src_T, src_min, src_k);
		Ez[ind_ijk] -= src_t * Jz[ind_src].real * b;
	}
}

void fdtd::FDTD::update_E(double t)
{
	dim3 fields_threadsPerBlock(8, 8, 8);
	dim3 fields_numBlocks(ceil((float)_Nx/fields_threadsPerBlock.x),
						  ceil((float)_Ny/fields_threadsPerBlock.y),
						  ceil((float)_Nz/fields_threadsPerBlock.z));

	update_E_fields <<<fields_numBlocks, fields_threadsPerBlock>>>
		(t, _R,
		 _Nx, _Ny, _Nz,
		 _dx, _dy, _dz, _dt,
		 _bc[0], _bc[1], _bc[2],
		 _w_pml_x0, _w_pml_x1,
		 _w_pml_y0, _w_pml_y1,
		 _w_pml_z0, _w_pml_z1,
		 _Ex, _Ey, _Ez,
		 _Hx, _Hy, _Hz,
		 _eps_x, _eps_y, _eps_z,
		 _pml_Hxy0, _pml_Hxy1, _pml_Hxz0, _pml_Hxz1,
		 _pml_Hyx0, _pml_Hyx1, _pml_Hyz0, _pml_Hyz1,
		 _pml_Hzx0, _pml_Hzx1, _pml_Hzy0, _pml_Hzy1,
		 _kappa_E_x, _kappa_E_y, _kappa_E_z,
		 _bEx, _bEy, _bEz,
		 _cEx, _cEy, _cEz);

    // Update sources
    for(auto const& src : _sources) {
		dim3 sources_threadsPerBlock(8, 8, 8);
		dim3 sources_numBlocks(ceil((float) src.K/sources_threadsPerBlock.x),
							   ceil((float) src.J/sources_threadsPerBlock.y),
							   ceil((float) src.I/sources_threadsPerBlock.z));

		update_E_sources <<<sources_numBlocks, sources_threadsPerBlock>>>
			(t,
			 _Nx, _Ny, _Nz,
			 _dt,
			 _Ex, _Ey, _Ez,
			 _eps_x, _eps_y, _eps_z,
			 _src_T, _src_min, _src_k,
			 src.i0, src.j0, src.k0,
			 src.I, src.J, src.K,
			 src.Jx, src.Jy, src.Jz);
		cudaDeviceSynchronize();
	}
}

///////////////////////////////////////////////////////////////////////////
// PML Management
///////////////////////////////////////////////////////////////////////////


void fdtd::FDTD::set_pml_widths(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    _w_pml_x0 = xmin; _w_pml_x1 = xmax;
    _w_pml_y0 = ymin; _w_pml_y1 = ymax;
    _w_pml_z0 = zmin; _w_pml_z1 = zmax;
}

void fdtd::FDTD::set_pml_properties(double sigma, double alpha, double kappa, double pow)
{
    _sigma = sigma;
    _alpha = alpha;
    _kappa = kappa;
    _pow   = pow;

    compute_pml_params();
}

void fdtd::FDTD::build_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(0 < xmin) {
        N = _Nz * _Ny * xmin;

        // Clean up old arrays and allocate new ones
        cudaFree(_pml_Eyx0);
        cudaFree(_pml_Ezx0);
        cudaMallocManaged((void **)&_pml_Eyx0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Ezx0, N*sizeof(double));

        cudaFree(_pml_Hyx0);
        cudaFree(_pml_Hzx0);
        cudaMallocManaged((void **)&_pml_Hyx0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hzx0, N*sizeof(double));
    }

    // touches xmax boundary
    if(_Nx > xmax) {
        N = _Nz * _Ny * (_Nx - xmax);

        // Clean up old arrays and allocate new ones
        cudaFree(_pml_Eyx1);
        cudaFree(_pml_Ezx1);
        cudaMallocManaged((void **)&_pml_Eyx1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Ezx1, N*sizeof(double));

        cudaFree(_pml_Hyx1);
        cudaFree(_pml_Hzx1);
        cudaMallocManaged((void **)&_pml_Hyx1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hzx1, N*sizeof(double));
    }

    // touches ymin boundary
    if(0 < ymin) {
        N = _Nz * _Nx * ymin;

        cudaFree(_pml_Exy0);
		cudaFree(_pml_Ezy0);
        cudaMallocManaged((void **)&_pml_Exy0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Ezy0, N*sizeof(double));

        cudaFree(_pml_Hxy0);
        cudaFree(_pml_Hzy0);
        cudaMallocManaged((void **)&_pml_Hxy0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hzy0, N*sizeof(double));
    }

    // touches ymax boundary
    if(_Ny > ymax) {
        N = _Nz * _Nx * (_Ny - ymax);

        cudaFree(_pml_Exy1);
        cudaFree(_pml_Ezy1);
        cudaMallocManaged((void **)&_pml_Exy1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Ezy1, N*sizeof(double));

        cudaFree(_pml_Hxy1);
		cudaFree(_pml_Hzy1);
        cudaMallocManaged((void **)&_pml_Hxy1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hzy1, N*sizeof(double));
    }

    // touches zmin boundary
    if(0 < zmin) {
        N = _Ny * _Nx * zmin;

        cudaFree(_pml_Exz0);
        cudaFree(_pml_Eyz0);
        cudaMallocManaged((void **)&_pml_Exz0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Eyz0, N*sizeof(double));

        cudaFree(_pml_Hxz0);
        cudaFree(_pml_Hyz0);
        cudaMallocManaged((void **)&_pml_Hxz0, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hyz0, N*sizeof(double));
    }

    // touches zmax boundary
    if(_Nz > zmax) {
        N = _Ny * _Nx * (_Nz - zmax);

        cudaFree(_pml_Exz1);
        cudaFree(_pml_Eyz1);
        cudaMallocManaged((void **)&_pml_Exz1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Eyz1, N*sizeof(double));

        cudaFree(_pml_Hxz1);
		cudaFree(_pml_Hyz1);
        cudaMallocManaged((void **)&_pml_Hxz1, N*sizeof(double));
        cudaMallocManaged((void **)&_pml_Hyz1, N*sizeof(double));
    }

    // (re)compute the spatially-dependent PML parameters
    compute_pml_params();
}

void fdtd::FDTD::reset_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,
        zmin = _w_pml_z0, zmax = _Nz-_w_pml_z1;

    // touches xmin boudary
    if(0 < xmin) {
        N = _Nz * _Ny * xmin;
        std::fill(_pml_Eyx0, _pml_Eyx0 + N, 0);
        std::fill(_pml_Ezx0, _pml_Ezx0 + N, 0);
        std::fill(_pml_Hyx0, _pml_Hyx0 + N, 0);
        std::fill(_pml_Hzx0, _pml_Hzx0 + N, 0);
    }

    // touches xmax boundary
    if(0 +_Nx > xmax) {
        N = _Nz * _Ny * (_Nx - xmax);

        std::fill(_pml_Eyx1, _pml_Eyx1 + N, 0);
        std::fill(_pml_Ezx1, _pml_Ezx1 + N, 0);
        std::fill(_pml_Hyx1, _pml_Hyx1 + N, 0);
        std::fill(_pml_Hzx1, _pml_Hzx1 + N, 0);
    }

    // touches ymin boundary
    if(0 < ymin) {
        N = _Nz * _Nx * ymin;

        std::fill(_pml_Exy0, _pml_Exy0 + N, 0);
        std::fill(_pml_Ezy0, _pml_Ezy0 + N, 0);
        std::fill(_pml_Hxy0, _pml_Hxy0 + N, 0);
        std::fill(_pml_Hzy0, _pml_Hzy0 + N, 0);
    }

    // touches ymax boundary
    if(_Ny > ymax) {
        N = _Nz * _Nx * (_Ny - ymax);

        std::fill(_pml_Exy1, _pml_Exy1 + N, 0);
        std::fill(_pml_Ezy1, _pml_Ezy1 + N, 0);
        std::fill(_pml_Hxy1, _pml_Hxy1 + N, 0);
        std::fill(_pml_Hzy1, _pml_Hzy1 + N, 0);
    }

    // touches zmin boundary
    if(0 < zmin) {
        N = _Ny * _Nx * zmin;

        std::fill(_pml_Exz0, _pml_Exz0 + N, 0);
        std::fill(_pml_Eyz0, _pml_Eyz0 + N, 0);
        std::fill(_pml_Hxz0, _pml_Hxz0 + N, 0);
        std::fill(_pml_Hyz0, _pml_Hyz0 + N, 0);
    }

    // touches zmax boundary
    if(_Nz > zmax) {
        N = _Ny * _Nx * (_Nz - zmax);

        std::fill(_pml_Exz1, _pml_Exz1 + N, 0);
        std::fill(_pml_Eyz1, _pml_Eyz1 + N, 0);
        std::fill(_pml_Hxz1, _pml_Hxz1 + N, 0);
        std::fill(_pml_Hyz1, _pml_Hyz1 + N, 0);
    }

}

void fdtd::FDTD::compute_pml_params()
{
    double pml_dist, pml_factor, sigma, alpha, kappa, b, c;

    // clean up the previous arrays and allocate new ones
    cudaFree(_kappa_H_x); cudaMallocManaged((void **)&_kappa_H_x, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_kappa_H_y); cudaMallocManaged((void **)&_kappa_H_y, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_kappa_H_z); cudaMallocManaged((void **)&_kappa_H_z, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

    cudaFree(_kappa_E_x); cudaMallocManaged((void **)&_kappa_E_x, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_kappa_E_y); cudaMallocManaged((void **)&_kappa_E_y, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_kappa_E_z); cudaMallocManaged((void **)&_kappa_E_z, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

    cudaFree(_bHx); cudaMallocManaged((void **)&_bHx, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_bHy); cudaMallocManaged((void **)&_bHy, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_bHz); cudaMallocManaged((void **)&_bHz, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

    cudaFree(_bEx); cudaMallocManaged((void **)&_bEx, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_bEy); cudaMallocManaged((void **)&_bEy, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_bEz); cudaMallocManaged((void **)&_bEz, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

    cudaFree(_cHx); cudaMallocManaged((void **)&_cHx, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_cHy); cudaMallocManaged((void **)&_cHy, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_cHz); cudaMallocManaged((void **)&_cHz, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

    cudaFree(_cEx); cudaMallocManaged((void **)&_cEx, sizeof(double)*(_w_pml_x0 + _w_pml_x1));
    cudaFree(_cEy); cudaMallocManaged((void **)&_cEy, sizeof(double)*(_w_pml_y0 + _w_pml_y1));
    cudaFree(_cEz); cudaMallocManaged((void **)&_cEz, sizeof(double)*(_w_pml_z0 + _w_pml_z1));

	std::cout << "after malloc, _cEz: " << _cEz << std::endl;	

    // calculate the PML parameters. These parameters are all functions of
    // the distance from the ONSET of the PML edge (which begins in the simulation
    // domain interior.
    // Note: PML parameters are ordered such that distance from PML onset
    // always increases with index.

    // setup xmin PML parameters
    for(int k = 0; k < _w_pml_x0; k++) {
        pml_dist = double(k - 0.5)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        // compute H coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1.0) * pml_factor+1.0;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[k] = kappa;
        _bHx[k] = b;
        _cHx[k] = c;

        pml_dist = double(k)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        // compute E coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1) * pml_factor+1;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[k] = kappa;
        _bEx[k] = b;
        _cEx[k] = c;

    }
    for(int k = 0; k < _w_pml_x1; k++) {
        // compute H coefficients
        pml_dist = double(k + 0.5)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[_w_pml_x0 + k] = kappa;
        _bHx[_w_pml_x0 + k] = b;
        _cHx[_w_pml_x0 + k] = c;

        //compute E coefficients
        pml_dist = double(k)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[_w_pml_x0 + k] = kappa;
        _bEx[_w_pml_x0 + k] = b;
        _cEx[_w_pml_x0 + k] = c;
    }
    for(int j = 0; j < _w_pml_y0; j++) {
        // calc H coefficients
        pml_dist = double(j - 0.5)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[j] = kappa;
        _bHy[j] = b;
        _cHy[j] = c;

        // calc E coefficients
        pml_dist = double(j)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_y[j] = kappa;
        _bEy[j] = b;
        _cEy[j] = c;

    }
    for(int j = 0; j < _w_pml_y1; j++) {
         // calc H coeffs
         pml_dist = double(j + 0.5)/_w_pml_y1; // distance from pml edge
         pml_factor = pml_ramp(pml_dist);

         sigma = _sigma * pml_factor;
         kappa = (_kappa-1) * pml_factor+1;
         alpha = _alpha * (1-pml_factor);
         b = exp(-_dt*(sigma/kappa + alpha));
         if(b == 1) c = 0;
         else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[_w_pml_y0 + j] = kappa;
        _bHy[_w_pml_y0 + j] = b;
        _cHy[_w_pml_y0 + j] = c;

        // compute E coefficients
        pml_dist = double(j)/_w_pml_y1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_y[_w_pml_y0 + j] = kappa;
        _bEy[_w_pml_y0 + j] = b;
        _cEy[_w_pml_y0 + j] = c;
    }

    for(int i = 0; i < _w_pml_z0; i++) {
        // calc H coeffs
        pml_dist = double(i)/_w_pml_z0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c= 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_z[i] = kappa;
        _bHz[i] = b;
        _cHz[i] = c;

        // calc E coeffs
        pml_dist = double(i+0.5)/_w_pml_z0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        // compute coefficients
        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_z[i] = kappa;
        _bEz[i] = b;
        _cEz[i] = c;
    }

    for(int i = 0; i < _w_pml_z1; i++) {
        // calc H coeffs
        pml_dist = double(i)/_w_pml_z1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_z[_w_pml_z0 + i] = kappa;
        _bHz[_w_pml_z0 + i] = b;
        _cHz[_w_pml_z0 + i] = c;

        // calc E coeffs
        pml_dist = double(i - 0.5)/_w_pml_z1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        // compute coefficients
        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_z[_w_pml_z0 + i] = kappa;
        _bEz[_w_pml_z0 + i] = b;
        _cEz[_w_pml_z0 + i] = c;
    }
}

double fdtd::FDTD::pml_ramp(double pml_dist)
{
    return std::pow(pml_dist, _pow);
}

///////////////////////////////////////////////////////////////////////////
// Amp/Phase Calculation management Management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_t0_arrays(complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                                complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0)
{
    _Ex_t0 = Ex_t0; _Ey_t0 = Ey_t0; _Ez_t0 = Ez_t0;
    _Hx_t0 = Hx_t0; _Hy_t0 = Hy_t0; _Hz_t0 = Hz_t0;
}

void fdtd::FDTD::set_t1_arrays(complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1)
{
_Ex_t1 = Ex_t1; _Ey_t1 = Ey_t1; _Ez_t1 = Ez_t1;
_Hx_t1 = Hx_t1; _Hy_t1 = Hy_t1; _Hz_t1 = Hz_t1;
}

void fdtd::FDTD::capture_t0_fields()
{
    int ind_ijk;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_ijk = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t0[ind_ijk] = _Ex[ind_ijk];
                _Ey_t0[ind_ijk] = _Ey[ind_ijk];
                _Ez_t0[ind_ijk] = _Ez[ind_ijk];

                _Hx_t0[ind_ijk] = _Hx[ind_ijk];
                _Hy_t0[ind_ijk] = _Hy[ind_ijk];
                _Hz_t0[ind_ijk] = _Hz[ind_ijk];
            }
        }
    }

}

void fdtd::FDTD::capture_t1_fields()
{
    int ind_ijk;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_ijk = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t1[ind_ijk] = _Ex[ind_ijk];
                _Ey_t1[ind_ijk] = _Ey[ind_ijk];
                _Ez_t1[ind_ijk] = _Ez[ind_ijk];

                _Hx_t1[ind_ijk] = _Hx[ind_ijk];
                _Hy_t1[ind_ijk] = _Hy[ind_ijk];
                _Hz_t1[ind_ijk] = _Hz[ind_ijk];
            }
        }
    }

}

void fdtd::FDTD::calc_complex_fields(double t0, double t1)
{
    double f0, f1, phi, A, t0H, t1H;
    int ind_ijk;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_ijk = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_ijk].real;
                f1 = _Ex[ind_ijk];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_ijk].real = A*cos(phi);
                _Ex_t0[ind_ijk].imag = -A*sin(phi);

                // Ey
                f0 = _Ey_t0[ind_ijk].real;
                f1 = _Ey[ind_ijk];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_ijk].real = A*cos(phi);
                _Ey_t0[ind_ijk].imag = -A*sin(phi);

                // Ez
                f0 = _Ez_t0[ind_ijk].real;
                f1 = _Ez[ind_ijk];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_ijk].real = A*cos(phi);
                _Ez_t0[ind_ijk].imag = -A*sin(phi);

                // Hx
                f0 = _Hx_t0[ind_ijk].real;
                f1 = _Hx[ind_ijk];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_ijk].real = A*cos(phi);
                _Hx_t0[ind_ijk].imag = -A*sin(phi);

                // Hy
                f0 = _Hy_t0[ind_ijk].real;
                f1 = _Hy[ind_ijk];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_ijk].real = A*cos(phi);
                _Hy_t0[ind_ijk].imag = -A*sin(phi);

                // Hz
                f0 = _Hz_t0[ind_ijk].real;
                f1 = _Hz[ind_ijk];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_ijk].real = A*cos(phi);
                _Hz_t0[ind_ijk].imag = -A*sin(phi);
            }
        }
    }

}


void fdtd::FDTD::calc_complex_fields(double t0, double t1, double t2)
{
    double f0, f1, f2, phi, A, t0H, t1H, t2H;
    int ind_ijk;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;
    t2H = t2 - 0.5*_dt;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_ijk = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_ijk].real;
                f1 = _Ex_t1[ind_ijk].real;
                f2 = _Ex[ind_ijk];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_ijk].real = A*cos(phi);
                _Ex_t0[ind_ijk].imag = -A*sin(phi);

                // Ey
                f0 = _Ey_t0[ind_ijk].real;
                f1 = _Ey_t1[ind_ijk].real;
                f2 = _Ey[ind_ijk];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_ijk].real = A*cos(phi);
                _Ey_t0[ind_ijk].imag = -A*sin(phi);

                // Ez
                f0 = _Ez_t0[ind_ijk].real;
                f1 = _Ez_t1[ind_ijk].real;
                f2 = _Ez[ind_ijk];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_ijk].real = A*cos(phi);
                _Ez_t0[ind_ijk].imag = -A*sin(phi);

                // Hx
                f0 = _Hx_t0[ind_ijk].real;
                f1 = _Hx_t1[ind_ijk].real;
                f2 = _Hx[ind_ijk];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_ijk].real = A*cos(phi);
                _Hx_t0[ind_ijk].imag = -A*sin(phi);

                // Hy
                f0 = _Hy_t0[ind_ijk].real;
                f1 = _Hy_t1[ind_ijk].real;
                f2 = _Hy[ind_ijk];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_ijk].real = A*cos(phi);
                _Hy_t0[ind_ijk].imag = -A*sin(phi);

                // Hz
                f0 = _Hz_t0[ind_ijk].real;
                f1 = _Hz_t1[ind_ijk].real;
                f2 = _Hz[ind_ijk];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_ijk].real = A*cos(phi);
                _Hz_t0[ind_ijk].imag = -A*sin(phi);

            }
        }
    }
}

inline double fdtd::calc_phase(double t0, double t1, double f0, double f1)
{
    if(f0 == 0.0 and f1 == 0) {
        return 0.0;
    }
    else {
        return atan((f1*sin(t0)-f0*sin(t1))/(f0*cos(t1)-f1*cos(t0)));
    }
}

inline double fdtd::calc_amplitude(double t0, double t1, double f0, double f1, double phase)
{
    if(f0*f0 > f1*f1) {
        return f1 / (sin(t1)*cos(phase) + cos(t1)*sin(phase));
    }
    else {
        return f0 / (sin(t0)*cos(phase) + cos(t0)*sin(phase));
    }
}

inline double fdtd::calc_phase(double t0, double t1, double t2, double f0, double f1, double f2)
{
    double f10 = f1 - f0,
           f21 = f2 - f1;

    if(f10 == 0 && f21 == 0) {
        return 0.0;
    }
    else {
        return atan2(f10*(sin(t2)-sin(t1)) - f21*(sin(t1)-sin(t0)),
                     f21*(cos(t1)-cos(t0)) - f10*(cos(t2)-cos(t1)));
    }
}

inline double fdtd::calc_amplitude(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    double f21 = f2 - f1,
           f10 = f1 - f0;

    if(f21 == 0 && f10 == 0) {
        return 0.0;
    }
    else if(f21*f21 >= f10*f10) {
        return f21 / (cos(phase)*(sin(t2)-sin(t1)) + sin(phase)*(cos(t2)-cos(t1)));
    }
    else {
        return f10 / (cos(phase)*(sin(t1)-sin(t0)) + sin(phase)*(cos(t1)-cos(t0)));
    }
}

///////////////////////////////////////////////////////////////////////////
// Source management
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::add_source(complex128 *Jx, complex128 *Jy, complex128 *Jz,
                            complex128 *Mx, complex128 *My, complex128 *Mz,
                            int i0, int j0, int k0, int I, int J, int K,
                            bool calc_phase)
{

    // these source arrays may *actually* be compelx-valued. In the time
    // domain, complex values correspond to temporal phase shifts. We need
    // to convert the complex value to an amplitude and phase. Fortunately,
    // we can use the memory that is already allocated for these values.
    // Specifically, we use src_array.real = amplitude and
    // src_array.imag = phase
    //
    // Important note: EMopt assumes the time dependence is exp(-i*omega*t).
    // In order to account for this minus sign, we need to invert the sign
    // of the calculated phase.
    if(calc_phase) {
		int ind=0;
		double real, imag;

		for(int i = 0; i < I; i++) {
			for(int j = 0; j < J; j++) {
				for(int k = 0; k < K; k++) {
					ind = i*J*K + j*K + k;


					// Jx
					real = Jx[ind].real;
					imag = Jx[ind].imag;

					Jx[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) Jx[ind].imag = 0.0;
					else Jx[ind].imag = -1*atan2(imag, real);

					// Jy
					real = Jy[ind].real;
					imag = Jy[ind].imag;

					Jy[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) Jy[ind].imag = 0.0;
					else Jy[ind].imag = -1*atan2(imag, real);

					// Jz
					real = Jz[ind].real;
					imag = Jz[ind].imag;

					Jz[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) Jz[ind].imag = 0.0;
					else Jz[ind].imag = -1*atan2(imag, real);

					// Mx
					real = Mx[ind].real;
					imag = Mx[ind].imag;

					Mx[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) Mx[ind].imag = 0.0;
					else Mx[ind].imag = -1*atan2(imag, real);

					// My
					real = My[ind].real;
					imag = My[ind].imag;

					My[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) My[ind].imag = 0.0;
					else My[ind].imag = -1*atan2(imag, real);

					// Mz
					real = Mz[ind].real;
					imag = Mz[ind].imag;

					Mz[ind].real = sqrt(real*real + imag*imag);
					if(imag == 0 && real == 0) Mz[ind].imag = 0.0;
					else Mz[ind].imag = -1*atan2(imag, real);

				}
			}
		}
    }
	else {
		int N = I * J * K;
		complex128 *cuda_Jx, *cuda_Jy, *cuda_Jz, *cuda_Mx, *cuda_My, *cuda_Mz;
		cudaMallocManaged((void **)&cuda_Jx, N*sizeof(complex128));
		cudaMallocManaged((void **)&cuda_Jy, N*sizeof(complex128));
		cudaMallocManaged((void **)&cuda_Jz, N*sizeof(complex128));
		cudaMallocManaged((void **)&cuda_Mx, N*sizeof(complex128));
		cudaMallocManaged((void **)&cuda_My, N*sizeof(complex128));
		cudaMallocManaged((void **)&cuda_Mz, N*sizeof(complex128));
		memcpy(cuda_Jx, Jx, N*sizeof(complex128));
		memcpy(cuda_Jy, Jy, N*sizeof(complex128));
		memcpy(cuda_Jz, Jz, N*sizeof(complex128));
		memcpy(cuda_Mx, Mx, N*sizeof(complex128));
		memcpy(cuda_My, My, N*sizeof(complex128));
		memcpy(cuda_Mz, Mz, N*sizeof(complex128));

		SourceArray src = {cuda_Jx, cuda_Jy, cuda_Jz, cuda_Mx, cuda_My, cuda_Mz, i0, j0, k0, I, J, K};
		_sources.push_back(src);
	}
}

void fdtd::FDTD::clear_sources()
{
    for(auto const& src : _sources) {
		cudaFree(src.Jx);
		cudaFree(src.Jy);
		cudaFree(src.Jz);
		cudaFree(src.Mx);
		cudaFree(src.My);
		cudaFree(src.Mz);
	}
    _sources.clear();
}

void fdtd::FDTD::set_source_properties(double src_T, double src_min)
{
    _src_T = src_T;
    _src_min = src_min;
    _src_k = src_T*src_T / log((1+src_min)/src_min);
    //_src_k = 6.0 / src_T; // rate of src turn on
    //_src_n0 = 1.0 / _src_k * log((1.0-src_min)/src_min); // src delay
}

inline double fdtd::FDTD::src_func_t(double t, double phase)
{
    if(t <= _src_T)
        return sin(t + phase)*((1+_src_min) * exp(-(t-_src_T)*(t-_src_T) / _src_k) - _src_min);
    else
        return sin(t + phase);
}


///////////////////////////////////////////////////////////////////////////
// Boundary Conditions
///////////////////////////////////////////////////////////////////////////
void fdtd::FDTD::set_bc(char* newbc)
{
    for(int i = 0; i < 3; i++){
        _bc[i] = newbc[i];
    }
}

///////////////////////////////////////////////////////////////////////////
// ctypes interface
///////////////////////////////////////////////////////////////////////////

fdtd::FDTD* FDTD_new(int Nx, int Ny, int Nz)
{
    return new fdtd::FDTD(Nx, Ny, Nz);
}

void FDTD_set_wavelength(fdtd::FDTD* fdtd, double wavelength)
{
    fdtd->set_wavelength(wavelength);
}

void FDTD_set_physical_dims(fdtd::FDTD* fdtd,
                            double X, double Y, double Z,
                            double dx, double dy, double dz)
{
    fdtd->set_physical_dims(X, Y, Z, dx, dy, dz);
}

void FDTD_set_dt(fdtd::FDTD* fdtd, double dt)
{
    fdtd->set_dt(dt);
}

void FDTD_set_complex_eps(fdtd::FDTD* fdtd, bool complex_eps)
{
    fdtd->set_complex_eps(complex_eps);
}

void FDTD_update_H(fdtd::FDTD* fdtd, double t)
{
    fdtd->update_H(t);
}

void FDTD_update_E(fdtd::FDTD* fdtd, double t)
{
    fdtd->update_E(t);
}

void FDTD_set_pml_widths(fdtd::FDTD* fdtd, int xmin, int xmax,
                                           int ymin, int ymax,
                                           int zmin, int zmax)
{
    fdtd->set_pml_widths(xmin, xmax, ymin, ymax, zmin, zmax);
}

void FDTD_set_pml_properties(fdtd::FDTD* fdtd, double sigma, double alpha,
                                               double kappa, double pow)
{
    fdtd->set_pml_properties(sigma, alpha, kappa, pow);
}

void FDTD_build_pml(fdtd::FDTD* fdtd)
{
    fdtd->build_pml();
}

void FDTD_reset_pml(fdtd::FDTD* fdtd)
{
    fdtd->reset_pml();
}

void FDTD_set_t0_arrays(fdtd::FDTD* fdtd,
                         complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                         complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0)
{
    fdtd->set_t0_arrays(Ex_t0, Ey_t0, Ez_t0, Hx_t0, Hy_t0, Hz_t0);
}

void FDTD_set_t1_arrays(fdtd::FDTD* fdtd,
                         complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
                         complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1)
{
    fdtd->set_t1_arrays(Ex_t1, Ey_t1, Ez_t1, Hx_t1, Hy_t1, Hz_t1);
}

double FDTD_calc_phase_2T(double t0, double t1, double f0, double f1)
{
    return fdtd::calc_phase(t0, t1, f0, f1);
}

double FDTD_calc_amplitude_2T(double t0, double t1, double f0, double f1, double phase)
{
    return fdtd::calc_amplitude(t0, t1, f0, f1, phase);
}

double FDTD_calc_phase_3T(double t0, double t1, double t2, double f0, double f1, double f2)
{
    return fdtd::calc_phase(t0, t1, t2, f0, f1, f2);
}

double FDTD_calc_amplitude_3T(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    return fdtd::calc_amplitude(t0, t1, t2, f0, f1, f2, phase);
}

void FDTD_capture_t0_fields(fdtd::FDTD* fdtd)
{
    fdtd->capture_t0_fields();
}

void FDTD_capture_t1_fields(fdtd::FDTD* fdtd)
{
    fdtd->capture_t1_fields();
}


void FDTD_calc_complex_fields_2T(fdtd::FDTD* fdtd, double t0, double t1)
{
    fdtd->calc_complex_fields(t0, t1);
}

void FDTD_calc_complex_fields_3T(fdtd::FDTD* fdtd, double t0, double t1, double t2)
{
    fdtd->calc_complex_fields(t0, t1, t2);
}

void FDTD_add_source(fdtd::FDTD* fdtd,
                     complex128 *Jx, complex128 *Jy, complex128 *Jz,
                     complex128 *Mx, complex128 *My, complex128 *Mz,
                     int i0, int j0, int k0, int I, int J, int K, bool calc_phase)
{
    fdtd->add_source(Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K, calc_phase);
}

void FDTD_clear_sources(fdtd::FDTD* fdtd)
{
    fdtd->clear_sources();
}

void FDTD_set_source_properties(fdtd::FDTD* fdtd, double src_T, double src_min)
{
    fdtd->set_source_properties(src_T, src_min);
}

double FDTD_src_func_t(fdtd::FDTD* fdtd, double t, double phase)
{
    return fdtd->src_func_t(t, phase);
}

void FDTD_set_bc(fdtd::FDTD* fdtd, char* newbc)
{
    fdtd->set_bc(newbc);
}
