#include "fdtd_cuda.hpp"
#include <math.h>
#include <algorithm>
#undef NDEBUG
#include <assert.h>

fdtd::FDTD::FDTD()
{
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
    // Clean up PML arrays
    delete[] _pml_Exy0; delete[] _pml_Exy1; delete[] _pml_Exz0; delete[] _pml_Exz1;
    delete[] _pml_Eyx0; delete[] _pml_Eyx1; delete[] _pml_Eyz0; delete[] _pml_Eyz1;
    delete[] _pml_Ezx0; delete[] _pml_Ezx1; delete[] _pml_Ezy0; delete[] _pml_Ezy1;
    delete[] _pml_Hxy0; delete[] _pml_Hxy1; delete[] _pml_Hxz0; delete[] _pml_Hxz1;
    delete[] _pml_Hyx0; delete[] _pml_Hyx1; delete[] _pml_Hyz0; delete[] _pml_Hyz1;
    delete[] _pml_Hzx0; delete[] _pml_Hzx1; delete[] _pml_Hzy0; delete[] _pml_Hzy1;

    delete [] _kappa_H_x;
    delete [] _kappa_H_y;
    delete [] _kappa_H_z;

    delete [] _kappa_E_x;
    delete [] _kappa_E_y;
    delete [] _kappa_E_z;

    delete [] _bHx;
    delete [] _bHy;
    delete [] _bHz;

    delete [] _bEx;
    delete [] _bEy;
    delete [] _bEz;

    delete [] _cHx;
    delete [] _cHy;
    delete [] _cHz;

    delete [] _cEx;
    delete [] _cEy;
    delete [] _cEz;
}

void fdtd::FDTD::set_physical_dims(double X, double Y, double Z,
                                         double dx, double dy, double dz)
{
    _X = X; _Y = Y; _Z = Z;
    _dx = dx; _dy = dy; _dz = dz;
}

void fdtd::FDTD::set_grid_dims(int Nx, int Ny, int Nz)
{
    _Nx = Nx;
    _Ny = Ny;
    _Nz = Nz;
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

void fdtd::FDTD::set_field_arrays(double *Ex, double *Ey, double *Ez,
                                  double *Hx, double *Hy, double *Hz)
{
    _Ex = Ex; _Ey = Ey; _Ez = Ez;
    _Hx = Hx; _Hy = Hy; _Hz = Hz;
}

void fdtd::FDTD::set_mat_arrays(complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                                complex128 *mu_x, complex128 *mu_y, complex128 *mu_z)
{
    _eps_x = eps_x; _eps_y = eps_y; _eps_z = eps_z;
    _mu_x = mu_x; _mu_y = mu_y; _mu_z = mu_z;
}

void fdtd::FDTD::update_H(double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           odz = _R/_dz,
           b, C, kappa,
           src_t,
           dt_by_mux, dt_by_muy, dt_by_muz;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    int ind_ijk, ind_ip1jk, ind_ijp1k, ind_ijkp1,
        ind_pml, ind_src, i0s, j0s, k0s, Is, Js, Ks,
        ind_pml_param;

    double dExdy, dExdz, dEydx, dEydz, dEzdx, dEzdy;
    complex128 *Mx, *My, *Mz;

    for(int i = 0; i < _Nz; i++) {
        int kill_zwrap = (_bc[2] != 'P' && i == _Nz-1) ? 1 : 0;
		int ip1 = i==_Nz-1 ? 0 : i+1;
        for(int j = 0; j < _Ny; j++) {
			int kill_ywrap = (_bc[1] != 'P' && j == _Ny-1) ? 1 : 0;
			int jp1 = j==_Ny-1 ? 0 : j+1;
			for(int k = 0; k < _Nx; k++) {
				int kill_xwrap = (_bc[0] != 'P' && k == _Nx-1) ? 1 : 0;
				int kp1 = k==_Nx-1 ? 0 : k+1;

                ind_ijk =   (i+0)*(_Ny+0)*(_Nx+0) + (j+0)*(_Nx+0) + k+0;
                ind_ijp1k = (i+0)*(_Ny+0)*(_Nx+0) + (jp1)*(_Nx+0) + k+0;
                ind_ip1jk = (ip1)*(_Ny+0)*(_Nx+0) + (j+0)*(_Nx+0) + k+0;
                ind_ijkp1 = (i+0)*(_Ny+0)*(_Nx+0) + (j+0)*(_Nx+0) + kp1;

				assert(ind_ijk >= 0 && ind_ijk < _Nz*_Ny*_Nx);
				assert(ind_ip1jk >= 0 && ind_ip1jk < _Nz*_Ny*_Nx);
				assert(ind_ijp1k >= 0 && ind_ijp1k < _Nz*_Ny*_Nx);
				assert(ind_ijkp1 >= 0 && ind_ijkp1 < _Nz*_Ny*_Nx);

                // compute prefactors
                dt_by_mux = _dt/_mu_x[ind_ijk].real;
                dt_by_muy = _dt/_mu_y[ind_ijk].real;
                dt_by_muz = _dt/_mu_z[ind_ijk].real;

                // Update Hx
                dEzdy = ody * ((kill_ywrap ? 0 : _Ez[ind_ijp1k])  - _Ez[ind_ijk]);
                dEydz = odz * ((kill_zwrap ? 0 : _Ey[ind_ip1jk])  - _Ey[ind_ijk]);

                _Hx[ind_ijk] = _Hx[ind_ijk] +  dt_by_mux *
                               (dEydz - dEzdy);

                // update Hy
                dExdz = odz * ((kill_zwrap ? 0 : _Ex[ind_ip1jk]) - _Ex[ind_ijk]);
                dEzdx = odx * ((kill_xwrap ? 0 : _Ez[ind_ijkp1]) - _Ez[ind_ijk]);

                _Hy[ind_ijk] = _Hy[ind_ijk] + dt_by_muy *
                               (dEzdx - dExdz);

                // update Hz
                dEydx = odx * ((kill_xwrap ? 0 : _Ey[ind_ijkp1]) - _Ey[ind_ijk]);
                dExdy = ody * ((kill_ywrap ? 0 : _Ex[ind_ijp1k]) - _Ex[ind_ijk]);

                _Hz[ind_ijk] = _Hz[ind_ijk] + dt_by_muz *
                               (dExdy - dEydx);

                // Do PML updates
                if(k < pml_xmin) {
                    // get index in PML array
                    ind_pml = i*_Ny*(pml_xmin) +j*(pml_xmin) + k;

                    // get PML coefficients
                    ind_pml_param = pml_xmin - k - 1;
                    kappa = _kappa_H_x[ind_pml_param];
                    b = _bHx[ind_pml_param];
                    C = _cHx[ind_pml_param];

                    // Update PML convolution
                    _pml_Eyx0[ind_pml] = C * dEydx + b*_pml_Eyx0[ind_pml];
                    _pml_Ezx0[ind_pml] = C * dEzdx + b*_pml_Ezx0[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] - dt_by_muz * (_pml_Eyx0[ind_pml]-dEydx+dEydx/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + dt_by_muy * (_pml_Ezx0[ind_pml]-dEzdx+dEzdx/kappa);

                }
                else if(k  >= pml_xmax) {
                    ind_pml = i*_Ny*(_Nx - pml_xmax) + j*(_Nx - pml_xmax) + k - pml_xmax;

                    // get pml coefficients
                    ind_pml_param = k - pml_xmax + _w_pml_x0;
                    kappa = _kappa_H_x[ind_pml_param];
                    b = _bHx[ind_pml_param];
                    C = _cHx[ind_pml_param];

                    _pml_Eyx1[ind_pml] = C * dEydx + b*_pml_Eyx1[ind_pml];
                    _pml_Ezx1[ind_pml] = C * dEzdx + b*_pml_Ezx1[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] - dt_by_muz * (_pml_Eyx1[ind_pml]-dEydx+dEydx/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + dt_by_muy * (_pml_Ezx1[ind_pml]-dEzdx+dEzdx/kappa);
                }

                if(j < pml_ymin) {
                    ind_pml = i*pml_ymin*_Nx +j*_Nx + k;

                    // compute coefficients
                    ind_pml_param = pml_ymin - j - 1;
                    kappa = _kappa_H_y[ind_pml_param];
                    b = _bHy[ind_pml_param];
                    C = _cHy[ind_pml_param];

                    _pml_Exy0[ind_pml] = C * dExdy + b*_pml_Exy0[ind_pml];
                    _pml_Ezy0[ind_pml] = C * dEzdy + b*_pml_Ezy0[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] + dt_by_muz * (_pml_Exy0[ind_pml]-dExdy+dExdy/kappa);
                    _Hx[ind_ijk] = _Hx[ind_ijk] - dt_by_mux * (_pml_Ezy0[ind_pml]-dEzdy+dEzdy/kappa);
                }
                else if(j >= pml_ymax) {
                    ind_pml = i*(_Ny - pml_ymax)*_Nx +(j - pml_ymax)*_Nx + k;

                    // compute coefficients
                    ind_pml_param = j - pml_ymax + _w_pml_y0;
                    kappa = _kappa_H_y[ind_pml_param];
                    b = _bHy[ind_pml_param];
                    C = _cHy[ind_pml_param];

                    _pml_Exy1[ind_pml] = C * dExdy + b*_pml_Exy1[ind_pml];
                    _pml_Ezy1[ind_pml] = C * dEzdy + b*_pml_Ezy1[ind_pml];

                    _Hz[ind_ijk] = _Hz[ind_ijk] + dt_by_muz * (_pml_Exy1[ind_pml]-dExdy+dExdy/kappa);
                    _Hx[ind_ijk] = _Hx[ind_ijk] - dt_by_mux * (_pml_Ezy1[ind_pml]-dEzdy+dEzdy/kappa);
                }

                if(i < pml_zmin) {
                    ind_pml = i*_Ny*_Nx +j*_Nx + k;

                    // get coefficients
                    ind_pml_param = pml_zmin - i - 1;
                    kappa = _kappa_H_z[ind_pml_param];
                    b = _bHz[ind_pml_param];
                    C = _cHz[ind_pml_param];

                    _pml_Exz0[ind_pml] = C * dExdz + b*_pml_Exz0[ind_pml];
                    _pml_Eyz0[ind_pml] = C * dEydz + b*_pml_Eyz0[ind_pml];

                    _Hx[ind_ijk] = _Hx[ind_ijk] + dt_by_mux * (_pml_Eyz0[ind_pml]-dEydz+dEydz/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] - dt_by_muy * (_pml_Exz0[ind_pml]-dExdz+dExdz/kappa);
                }
                else if(i > pml_zmax) {
                    ind_pml = (i - pml_zmax)*_Ny*_Nx +j*_Nx + k;

                    // get coefficients
                    ind_pml_param = i - pml_zmax + _w_pml_z0;
                    kappa = _kappa_H_z[ind_pml_param];
                    b = _bHz[ind_pml_param];
                    C = _cHz[ind_pml_param];

                    _pml_Exz1[ind_pml] = C * dExdz + b*_pml_Exz1[ind_pml];
                    _pml_Eyz1[ind_pml] = C * dEydz + b*_pml_Eyz1[ind_pml];

                    _Hx[ind_ijk] = _Hx[ind_ijk] + dt_by_mux * (_pml_Eyz1[ind_pml]-dEydz+dEydz/kappa);
                    _Hy[ind_ijk] = _Hy[ind_ijk] - dt_by_muy * (_pml_Exz1[ind_pml]-dExdz+dExdz/kappa);
                }

            }
        }
    }

    // Update sources
    for(auto const& src : _sources) {
        i0s = src.i0; Is = src.I;
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Mx
        Mx = src.Mx;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

					assert(ind_ijk >= 0 && ind_ijk < _Nz*_Ny*_Nx);
					assert(ind_src >= 0 && ind_src < _Nz*_Ny*_Nx);

                    src_t = src_func_t(t, Mx[ind_src].imag);
                    _Hx[ind_ijk] = _Hx[ind_ijk] + src_t * Mx[ind_src].real * _dt / _mu_x[ind_ijk].real;
                }
            }
        }

        // update My
        My = src.My;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

					assert(ind_ijk >= 0 && ind_ijk < _Nz*_Ny*_Nx);
					assert(ind_src >= 0 && ind_src < _Nz*_Ny*_Nx);

                    src_t = src_func_t(t, My[ind_src].imag);
                    _Hy[ind_ijk] = _Hy[ind_ijk] + src_t * My[ind_src].real * _dt / _mu_y[ind_ijk].real;
                }
            }
        }

        // update Mz
        Mz = src.Mz;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

					assert(ind_ijk >= 0 && ind_ijk < _Nz*_Ny*_Nx);
					assert(ind_src >= 0 && ind_src < _Nz*_Ny*_Nx);

                    src_t = src_func_t(t, Mz[ind_src].imag);
                    _Hz[ind_ijk] = _Hz[ind_ijk] + src_t * Mz[ind_src].real * _dt / _mu_z[ind_ijk].real;
                }
            }
        }

    }
}

void fdtd::FDTD::update_E(double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           odz = _R/_dz,
           b, C, kappa,
           src_t,
           a_x, a_y, a_z, b_x, b_y, b_z;

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,
        pml_zmin = _w_pml_z0, pml_zmax = _Nz-_w_pml_z1;

    int ind_ijk, ind_im1jk, ind_ijm1k, ind_ijkm1,
        ind_pml, ind_src, i0s, j0s, k0s, Is, Js, Ks,
        ind_pml_param;

    double dHxdy, dHxdz, dHydx, dHydz, dHzdx, dHzdy;
    complex128 *Jx, *Jy, *Jz;
    enum action { ACTION_NOP, ACTION_ZERO, ACTION_FLIP, ACTION_COPY };

    for(int i = 0; i < _Nz; i++) {
        int action_zwrap = (_bc[2] == 'P' || i != 0 ? ACTION_NOP :
							_bc[2] == '0'           ? ACTION_ZERO :
							_bc[2] == 'E'           ? ACTION_FLIP : ACTION_COPY);
		int im1 = i==0 ? _Nz-1 : i-1;
        for(int j = 0; j < _Ny; j++) {
			int action_ywrap = (_bc[1] == 'P' || j != 0 ? ACTION_NOP :
								_bc[1] == '0'           ? ACTION_ZERO :
								_bc[1] == 'E'           ? ACTION_FLIP : ACTION_COPY);
			int jm1 = j==0 ? _Ny-1 : j-1;
			for(int k = 0; k < _Nx; k++) {
				int action_xwrap = (_bc[0] == 'P' || k != 0 ? ACTION_NOP :
									_bc[0] == '0'           ? ACTION_ZERO :
									_bc[0] == 'E'           ? ACTION_FLIP : ACTION_COPY);
				int km1 = k==0 ? _Nx-1 : k-1;

                ind_ijk   = (i-0)*(_Ny)*(_Nx) + (j-0)*(_Nx) + k-0;
                ind_ijm1k = (i-0)*(_Ny)*(_Nx) + (jm1)*(_Nx) + k-0;
                ind_im1jk = (im1)*(_Ny)*(_Nx) + (j-0)*(_Nx) + k-0;
                ind_ijkm1 = (i-0)*(_Ny)*(_Nx) + (j-0)*(_Nx) + km1;

				assert(ind_ijk >= 0 && ind_ijk < _Nz*_Ny*_Nx);
				assert(ind_im1jk >= 0 && ind_im1jk < _Nz*_Ny*_Nx);
				assert(ind_ijm1k >= 0 && ind_ijm1k < _Nz*_Ny*_Nx);
				assert(ind_ijkm1 >= 0 && ind_ijkm1 < _Nz*_Ny*_Nx);

                a_x = 1.0; a_y = 1.0; a_z = 1.0;
                b_x = _dt/_eps_x[ind_ijk].real;
                b_y = _dt/_eps_y[ind_ijk].real;
                b_z = _dt/_eps_z[ind_ijk].real;

                // Update Ex
                dHzdy = ody*(_Hz[ind_ijk] - (action_ywrap == ACTION_ZERO ? 0 :
					     action_ywrap == ACTION_COPY ?  _Hz[ind_ijk] :
					     action_ywrap == ACTION_FLIP ? -_Hz[ind_ijk] : _Hz[ind_ijm1k]));
                dHydz = odz*(_Hy[ind_ijk] - (action_zwrap == ACTION_ZERO ? 0 :
					     action_zwrap == ACTION_COPY ?  _Hy[ind_ijk] :
					     action_zwrap == ACTION_FLIP ? -_Hy[ind_ijk] : _Hy[ind_im1jk]));

                _Ex[ind_ijk] = a_x*_Ex[ind_ijk] + (dHzdy - dHydz) * b_x;

                // Update Ey
                dHxdz = odz*(_Hx[ind_ijk] - (action_zwrap == ACTION_ZERO ? 0 :
					     action_zwrap == ACTION_COPY ?  _Hx[ind_ijk] :
					     action_zwrap == ACTION_FLIP ? -_Hx[ind_ijk] : _Hx[ind_im1jk]));
                dHzdx = odx*(_Hz[ind_ijk] - (action_xwrap == ACTION_ZERO ? 0 :
					     action_xwrap == ACTION_COPY ?  _Hz[ind_ijk] :
					     action_xwrap == ACTION_FLIP ? -_Hz[ind_ijk] : _Hz[ind_ijkm1]));

                _Ey[ind_ijk] = a_y * _Ey[ind_ijk] + (dHxdz - dHzdx) * b_y;

                // Update Ez
                dHydx = odx*(_Hy[ind_ijk] - (action_xwrap == ACTION_ZERO ? 0 :
					     action_xwrap == ACTION_COPY ?  _Hy[ind_ijk] :
					     action_xwrap == ACTION_FLIP ? -_Hy[ind_ijk] : _Hy[ind_ijkm1]));
                dHxdy = ody*(_Hx[ind_ijk] - (action_ywrap == ACTION_ZERO ? 0 :
					     action_ywrap == ACTION_COPY ?  _Hx[ind_ijk] :
					     action_ywrap == ACTION_FLIP ? -_Hx[ind_ijk] : _Hx[ind_ijm1k]));

                _Ez[ind_ijk] = a_z * _Ez[ind_ijk] + (dHydx - dHxdy) * b_z;


                // Do PML updates
                if(k < pml_xmin) {
                    ind_pml = i*_Ny*(pml_xmin) +j*(pml_xmin) + k;

                    // get PML coefficients
                    ind_pml_param = pml_xmin - k - 1;
                    kappa = _kappa_E_x[ind_pml_param];
                    b = _bEx[ind_pml_param];
                    C = _cEx[ind_pml_param];

                    _pml_Hyx0[ind_pml] = C * dHydx + b*_pml_Hyx0[ind_pml];
                    _pml_Hzx0[ind_pml] = C * dHzdx + b*_pml_Hzx0[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] + (_pml_Hyx0[ind_pml]-dHydx+dHydx/kappa) * b_z;
                    _Ey[ind_ijk] = _Ey[ind_ijk] - (_pml_Hzx0[ind_pml]-dHzdx+dHzdx/kappa) * b_y;

                }
                else if(k >= pml_xmax) {
                    ind_pml = i*_Ny*(_Nx - pml_xmax) +j*(_Nx - pml_xmax) + k - pml_xmax;

                    // get coefficients
                    ind_pml_param = k - pml_xmax + _w_pml_x0;
                    kappa = _kappa_E_x[ind_pml_param];
                    b = _bEx[ind_pml_param];
                    C = _cEx[ind_pml_param];

                    _pml_Hyx1[ind_pml] = C * dHydx + b*_pml_Hyx1[ind_pml];
                    _pml_Hzx1[ind_pml] = C * dHzdx + b*_pml_Hzx1[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] + (_pml_Hyx1[ind_pml]-dHydx+dHydx/kappa) * b_z;
                    _Ey[ind_ijk] = _Ey[ind_ijk] - (_pml_Hzx1[ind_pml]-dHzdx+dHzdx/kappa) * b_y;
                }

                if(j < pml_ymin) {
                    ind_pml = i*pml_ymin*_Nx +j*_Nx + k;

                    // get coefficients
                    ind_pml_param = pml_ymin - j - 1;
                    kappa = _kappa_E_y[ind_pml_param];
                    b = _bEy[ind_pml_param];
                    C = _cEy[ind_pml_param];

                    _pml_Hxy0[ind_pml] = C * dHxdy + b*_pml_Hxy0[ind_pml];
                    _pml_Hzy0[ind_pml] = C * dHzdy + b*_pml_Hzy0[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] - (_pml_Hxy0[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
                    _Ex[ind_ijk] = _Ex[ind_ijk] + (_pml_Hzy0[ind_pml]-dHzdy+dHzdy/kappa) * b_x;
                }
                else if(j >= pml_ymax) {
                    ind_pml = i*(_Ny - pml_ymax)*_Nx +(j - pml_ymax)*_Nx + k;

                    // get coefficients
                    ind_pml_param = j - pml_ymax + _w_pml_y0;
                    kappa = _kappa_E_y[ind_pml_param];
                    b = _bEy[ind_pml_param];
                    C = _cEy[ind_pml_param];

                    _pml_Hxy1[ind_pml] = C * dHxdy + b*_pml_Hxy1[ind_pml];
                    _pml_Hzy1[ind_pml] = C * dHzdy + b*_pml_Hzy1[ind_pml];

                    _Ez[ind_ijk] = _Ez[ind_ijk] - (_pml_Hxy1[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
                    _Ex[ind_ijk] = _Ex[ind_ijk] + (_pml_Hzy1[ind_pml]-dHzdy+dHzdy/kappa) * b_x;
                }

                if(i < pml_zmin) {
                    ind_pml = i*_Ny*_Nx +j*_Nx + k;

                    // get coefficients
                    ind_pml_param = pml_zmin - i - 1;
                    kappa = _kappa_E_z[ind_pml_param];
                    b = _bEz[ind_pml_param];
                    C = _cEz[ind_pml_param];

                    _pml_Hxz0[ind_pml] = C * dHxdz + b*_pml_Hxz0[ind_pml];
                    _pml_Hyz0[ind_pml] = C * dHydz + b*_pml_Hyz0[ind_pml];

                    _Ex[ind_ijk] = _Ex[ind_ijk] - (_pml_Hyz0[ind_pml]-dHydz+dHydz/kappa) * b_x;
                    _Ey[ind_ijk] = _Ey[ind_ijk] + (_pml_Hxz0[ind_pml]-dHxdz+dHxdz/kappa) * b_y;
                }
                else if(i > pml_zmax) {
                    ind_pml = (i - pml_zmax)*_Ny*_Nx +j*_Nx + k;

                    // compute coefficients
                    ind_pml_param = i - pml_zmax + _w_pml_z0;
                    kappa = _kappa_E_z[ind_pml_param];
                    b = _bEz[ind_pml_param];
                    C = _cEz[ind_pml_param];

                    _pml_Hxz1[ind_pml] = C * dHxdz + b*_pml_Hxz1[ind_pml];
                    _pml_Hyz1[ind_pml] = C * dHydz + b*_pml_Hyz1[ind_pml];

                    _Ex[ind_ijk] = _Ex[ind_ijk] - (_pml_Hyz1[ind_pml]-dHydz+dHydz/kappa) * b_x;
                    _Ey[ind_ijk] = _Ey[ind_ijk] + (_pml_Hxz1[ind_pml]-dHxdz+dHxdz/kappa) * b_y;
                }
            }
        }
    }

    // Update sources
    for(auto const& src : _sources) {
        i0s = src.i0; Is = src.I;
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Jx
        Jx = src.Jx;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk    = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

                    b_x = _dt/_eps_x[ind_ijk].real;

                    src_t = src_func_t(t, Jx[ind_src].imag);
                    _Ex[ind_ijk] = _Ex[ind_ijk] - src_t * Jx[ind_src].real * b_x;
                }
            }
        }

        // update Jy
        Jy = src.Jy;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

                    b_y = _dt/_eps_y[ind_ijk].real;

                    src_t = src_func_t(t, Jy[ind_src].imag);
                    _Ey[ind_ijk] = _Ey[ind_ijk] - src_t * Jy[ind_src].real * b_y;
                }
            }
        }

        // update Jz
        Jz = src.Jz;

        for(int i = 0; i < Is; i++) {
            for(int j = 0; j < Js; j++) {
                for(int k = 0; k < Ks; k++) {
                    ind_ijk = (i+i0s)*(_Ny)*(_Nx) + (j+j0s)*(_Nx) + k + k0s;
                    ind_src = i*Js*Ks + j*Ks + k;

                    b_z = _dt/_eps_z[ind_ijk].real;

                    src_t = src_func_t(t, Jz[ind_src].imag);
                    _Ez[ind_ijk] = _Ez[ind_ijk] - src_t * Jz[ind_src].real * b_z;
                }
            }
        }
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
        delete [] _pml_Eyx0; _pml_Eyx0 = NULL;
        delete [] _pml_Ezx0; _pml_Ezx0 = NULL;
        _pml_Eyx0 = new double[N];
        _pml_Ezx0 = new double[N];

        delete [] _pml_Hyx0; _pml_Hyx0 = NULL;
        delete [] _pml_Hzx0; _pml_Hzx0 = NULL;
        _pml_Hyx0 = new double[N];
        _pml_Hzx0 = new double[N];
    }

    // touches xmax boundary
    if(_Nx > xmax) {
        N = _Nz * _Ny * (_Nx - xmax);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Eyx1; _pml_Eyx1 = NULL;
        delete [] _pml_Ezx1; _pml_Ezx1 = NULL;
        _pml_Eyx1 = new double[N];
        _pml_Ezx1 = new double[N];

        delete [] _pml_Hyx1; _pml_Hyx1 = NULL;
        delete [] _pml_Hzx1; _pml_Hzx1 = NULL;
        _pml_Hyx1 = new double[N];
        _pml_Hzx1 = new double[N];
    }

    // touches ymin boundary
    if(0 < ymin) {
        N = _Nz * _Nx * ymin;

        delete [] _pml_Exy0; _pml_Exy0 = NULL;
        delete [] _pml_Ezy0; _pml_Ezy0 = NULL;
        _pml_Exy0 = new double[N];
        _pml_Ezy0 = new double[N];

        delete [] _pml_Hxy0; _pml_Hxy0 = NULL;
        delete [] _pml_Hzy0; _pml_Hzy0 = NULL;
        _pml_Hxy0 = new double[N];
        _pml_Hzy0 = new double[N];
    }

    // touches ymax boundary
    if(_Ny > ymax) {
        N = _Nz * _Nx * (_Ny - ymax);

        delete [] _pml_Exy1; _pml_Exy1 = NULL;
        delete [] _pml_Ezy1; _pml_Ezy1 = NULL;
        _pml_Exy1 = new double[N];
        _pml_Ezy1 = new double[N];

        delete [] _pml_Hxy1; _pml_Hxy1 = NULL;
        delete [] _pml_Hzy1; _pml_Hzy1 = NULL;
        _pml_Hxy1 = new double[N];
        _pml_Hzy1 = new double[N];
    }

    // touches zmin boundary
    if(0 < zmin) {
        N = _Ny * _Nx * zmin;

        delete [] _pml_Exz0; _pml_Exz0 = NULL;
        delete [] _pml_Eyz0; _pml_Eyz0 = NULL;
        _pml_Exz0 = new double[N];
        _pml_Eyz0 = new double[N];

        delete [] _pml_Hxz0; _pml_Hxz0 = NULL;
        delete [] _pml_Hyz0; _pml_Hyz0 = NULL;
        _pml_Hxz0 = new double[N];
        _pml_Hyz0 = new double[N];
    }

    // touches zmax boundary
    if(_Nz > zmax) {
        N = _Ny * _Nx * (_Nz - zmax);

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Exz1 = new double[N];
        _pml_Eyz1 = new double[N];

        delete [] _pml_Hxz1; _pml_Hxz1 = NULL;
        delete [] _pml_Hyz1; _pml_Hyz1 = NULL;
        _pml_Hxz1 = new double[N];
        _pml_Hyz1 = new double[N];
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
    delete [] _kappa_H_x; _kappa_H_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_H_y; _kappa_H_y = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _kappa_H_z; _kappa_H_z = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _kappa_E_x; _kappa_E_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_E_y; _kappa_E_y = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _kappa_E_z; _kappa_E_z = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _bHx; _bHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bHy; _bHy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _bHz; _bHz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _bEx; _bEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bEy; _bEy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _bEz; _bEz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _cHx; _cHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cHy; _cHy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _cHz; _cHz = new double[_w_pml_z0 + _w_pml_z1];

    delete [] _cEx; _cEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cEy; _cEy = new double[_w_pml_y0 + _w_pml_y1];
    delete [] _cEz; _cEz = new double[_w_pml_z0 + _w_pml_z1];

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
    int ind_local;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_local = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t0[ind_local] = _Ex[ind_local];
                _Ey_t0[ind_local] = _Ey[ind_local];
                _Ez_t0[ind_local] = _Ez[ind_local];

                _Hx_t0[ind_local] = _Hx[ind_local];
                _Hy_t0[ind_local] = _Hy[ind_local];
                _Hz_t0[ind_local] = _Hz[ind_local];
            }
        }
    }

}

void fdtd::FDTD::capture_t1_fields()
{
    int ind_local;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_local = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Copy the fields at the current time to the auxillary arrays
                _Ex_t1[ind_local] = _Ex[ind_local];
                _Ey_t1[ind_local] = _Ey[ind_local];
                _Ez_t1[ind_local] = _Ez[ind_local];

                _Hx_t1[ind_local] = _Hx[ind_local];
                _Hy_t1[ind_local] = _Hy[ind_local];
                _Hz_t1[ind_local] = _Hz[ind_local];
            }
        }
    }

}

void fdtd::FDTD::calc_complex_fields(double t0, double t1)
{
    double f0, f1, phi, A, t0H, t1H;
    int ind_local;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_local = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_local].real;
                f1 = _Ex[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_local].real = A*cos(phi);
                _Ex_t0[ind_local].imag = -A*sin(phi);

                // Ey
                f0 = _Ey_t0[ind_local].real;
                f1 = _Ey[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_local].real = A*cos(phi);
                _Ey_t0[ind_local].imag = -A*sin(phi);

                // Ez
                f0 = _Ez_t0[ind_local].real;
                f1 = _Ez[ind_local];
                phi = calc_phase(t0, t1, f0, f1);
                A = calc_amplitude(t0, t1, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_local].real = A*cos(phi);
                _Ez_t0[ind_local].imag = -A*sin(phi);

                // Hx
                f0 = _Hx_t0[ind_local].real;
                f1 = _Hx[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_local].real = A*cos(phi);
                _Hx_t0[ind_local].imag = -A*sin(phi);

                // Hy
                f0 = _Hy_t0[ind_local].real;
                f1 = _Hy[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_local].real = A*cos(phi);
                _Hy_t0[ind_local].imag = -A*sin(phi);

                // Hz
                f0 = _Hz_t0[ind_local].real;
                f1 = _Hz[ind_local];
                phi = calc_phase(t0H, t1H, f0, f1);
                A = calc_amplitude(t0H, t1H, f0, f1, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_local].real = A*cos(phi);
                _Hz_t0[ind_local].imag = -A*sin(phi);
            }
        }
    }

}


void fdtd::FDTD::calc_complex_fields(double t0, double t1, double t2)
{
    double f0, f1, f2, phi, A, t0H, t1H, t2H;
    int ind_local;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;
    t2H = t2 - 0.5*_dt;

    for(int i = 0; i < _Nz; i++) {
        for(int j = 0; j < _Ny; j++) {
            for(int k = 0; k < _Nx; k++) {
                ind_local = (i)*(_Ny)*(_Nx) + (j)*(_Nx) + k;

                // Compute amplitude and phase for Ex
                // Note: we are careful to assume exp(-i*w*t) time dependence
                f0 = _Ex_t0[ind_local].real;
                f1 = _Ex_t1[ind_local].real;
                f2 = _Ex[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ex_t0[ind_local].real = A*cos(phi);
                _Ex_t0[ind_local].imag = -A*sin(phi);

                // Ey
                f0 = _Ey_t0[ind_local].real;
                f1 = _Ey_t1[ind_local].real;
                f2 = _Ey[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ey_t0[ind_local].real = A*cos(phi);
                _Ey_t0[ind_local].imag = -A*sin(phi);

                // Ez
                f0 = _Ez_t0[ind_local].real;
                f1 = _Ez_t1[ind_local].real;
                f2 = _Ez[ind_local];
                phi = calc_phase(t0, t1, t2, f0, f1, f2);
                A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Ez_t0[ind_local].real = A*cos(phi);
                _Ez_t0[ind_local].imag = -A*sin(phi);

                // Hx
                f0 = _Hx_t0[ind_local].real;
                f1 = _Hx_t1[ind_local].real;
                f2 = _Hx[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hx_t0[ind_local].real = A*cos(phi);
                _Hx_t0[ind_local].imag = -A*sin(phi);

                // Hy
                f0 = _Hy_t0[ind_local].real;
                f1 = _Hy_t1[ind_local].real;
                f2 = _Hy[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hy_t0[ind_local].real = A*cos(phi);
                _Hy_t0[ind_local].imag = -A*sin(phi);

                // Hz
                f0 = _Hz_t0[ind_local].real;
                f1 = _Hz_t1[ind_local].real;
                f2 = _Hz[ind_local];
                phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
                A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
                if(A < 0) {
                    A *= -1;
                    phi += M_PI;
                }
                _Hz_t0[ind_local].real = A*cos(phi);
                _Hz_t0[ind_local].imag = -A*sin(phi);

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
    int ind=0;
    double real, imag;
    SourceArray src = {Jx, Jy, Jz, Mx, My, Mz, i0, j0, k0, I, J, K};

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

    _sources.push_back(src);
}

void fdtd::FDTD::clear_sources()
{
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

fdtd::FDTD* FDTD_new()
{
    return new fdtd::FDTD();
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

void FDTD_set_grid_dims(fdtd::FDTD* fdtd, int Nx, int Ny, int Nz)
{
    fdtd->set_grid_dims(Nx, Ny, Nz);
}

void FDTD_set_dt(fdtd::FDTD* fdtd, double dt)
{
    fdtd->set_dt(dt);
}

void FDTD_set_complex_eps(fdtd::FDTD* fdtd, bool complex_eps)
{
    fdtd->set_complex_eps(complex_eps);
}

void FDTD_set_field_arrays(fdtd::FDTD* fdtd,
                           double *Ex, double *Ey, double *Ez,
                           double *Hx, double *Hy, double *Hz)
{
    fdtd->set_field_arrays(Ex, Ey, Ez, Hx, Hy, Hz);
}

void FDTD_set_mat_arrays(fdtd::FDTD* fdtd,
                         complex128 *eps_x, complex128 *eps_y, complex128 *eps_z,
                         complex128 *mu_x, complex128 *mu_y, complex128 *mu_z)
{
    fdtd->set_mat_arrays(eps_x, eps_y, eps_z, mu_x, mu_y, mu_z);
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
