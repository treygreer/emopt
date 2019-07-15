#include <iostream>
#include <memory>
#include <vector>

#ifndef __FDTD_CUDA_HPP__
#define __FDTD_CUDA_HPP__

#define pow2(x) x*x
#define pow3(x) x*x*x

/** A complex value data type that is compatible with numpy.
 *
 * Currently, the computationally intensive components of FDTD 
 * are written in C++ while the less intensive parts are written
 * in python. In order to facilitate moving data back and forth
 * between python and C++, we need to define a simple 128 bit 
 * complex data type (64 bit double precision for real + imag).
 */
typedef struct struct_complex128 {
    double real, imag;

    struct_complex128 operator+(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real + val.real;
        output.imag = 0;
        return output;
    }

    struct_complex128 operator-(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real - val.real;
        output.imag = 0;
        return output;
    }

    struct_complex128 operator*(const struct_complex128& val) {
        struct_complex128 output;
        output.real = real*val.real;
        output.imag =  0;
        return output;
    }

    struct_complex128 operator/(double val1) {
        struct_complex128 output;
        output.real = real/val1;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator/(double val1, const struct_complex128& val2) {
        struct_complex128 output;

        output.real = val1 / val2.real;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator*(double val1, const struct_complex128& val2) {
        struct_complex128 output;
        output.real = val1*val2.real;
        output.imag = 0;
        return output;
    }

    friend struct_complex128 operator+(double val1, const struct_complex128& val2) {
        struct_complex128 output;
        output.real = val1 + val2.real;
        output.imag = 0;
        return output;
    }

    struct_complex128& operator=(double val) {
        real = val;
		imag = 0;
        return *this;
    }

} complex128;

namespace fdtd {

    /*!
     * Calculates the phase of a sinusoidal signal using three data points.
     *
     * By using three data points, we can account for any DC offset in the
     * sinusoid. In other words, this solves for the phase \phi of the
     * function
     *
     *      f(t) = A*\sin(\omega t + \phi) + B
     *
     * \param t0 - The time of the first sample.
     * \param t1 - The time of the second sample.
     * \param t2 - The time of the third sample.
     * \param f0 - The first sample of the sinusoid.
     * \param f1 - The second sample of the sinusoid.
     * \param f2 - The third sample of the sinusoid.
     *
     * \return the phase of the sinusoid.
     */
    double calc_phase_3T(double t0, double t1, double t2, double f0, double f1, double f2);

    /*!
     * Calculates the amplitude of a sinusoidal signal using three data points.
     *
     * By using three data points, we can account for any DC offset in the
     * sinusoid. In other words, this solves for the amplitude A of the
     * function
     *
     *      f(t) = A*\sin(\omega t + \phi) + B
     *
     * \param t0 - The time of the first sample.
     * \param t1 - The time of the second sample.
     * \param t2 - The time of the third sample.
     * \param f0 - The first sample of the sinusoid.
     * \param f1 - The second sample of the sinusoid.
     * \param f2 - The third sample of the sinusoid.
     *
     * \return the amplitude of the sinusoid.
     */
    double calc_amplitude_3T(double t0, double t1, double t2,
							 double f0, double f1, double f2, double phase);

    typedef struct struct_SourceArray {
        complex128 *Jx, *Jy, *Jz, *Mx, *My, *Mz;
        int i0, j0, k0, I, J, K;
    } SourceArray;

	typedef struct struct_CudaData {
		// Field and source arrays
		double  *Ex, *Ey, *Ez,
			*Hx, *Hy, *Hz;

		// Material arrays
		complex128 *eps_x, *eps_y, *eps_z,
			*mu_x, *mu_y, *mu_z;

		// number of Yee cells in X, Y, Z
		int Nx, Ny, Nz;

		// physical simulation size and Yee cell size in x,y,z
		double dx, dy, dz;

		// R over dx,dy,dz
		double odx, ody, odz;

		// time step
		double dt;

		// source time parameters
		double src_T, src_min, src_k, src_n0;


		char bc0, bc1, bc2;

		// PML arrays -- because convolutions
		// Not ever processor will need all of different PML layers.
		// For example, a processor which touches the xmin boundary of the
		// simulation only needs to store pml values corresponding to derivatives
		// along the x direction.
        int pml_xmin, pml_xmax, pml_ymin, pml_ymax, pml_zmin, pml_zmax;
		double *pml_Exy0, *pml_Exy1, *pml_Exz0, *pml_Exz1,
			*pml_Eyx0, *pml_Eyx1, *pml_Eyz0, *pml_Eyz1,
			*pml_Ezx0, *pml_Ezx1, *pml_Ezy0, *pml_Ezy1,
			*pml_Hxy0, *pml_Hxy1, *pml_Hxz0, *pml_Hxz1,
			*pml_Hyx0, *pml_Hyx1, *pml_Hyz0, *pml_Hyz1,
			*pml_Hzx0, *pml_Hzx1, *pml_Hzy0, *pml_Hzy1;

		// precomputed pml parameters. These values are precomputed to speed things up
		double *kappa_H_x, *kappa_H_y, *kappa_H_z, 
			*kappa_E_x, *kappa_E_y, *kappa_E_z,
			*bHx, *bHy, *bHz,
			*bEx, *bEy, *bEz,
			*cHx, *cHy, *cHz,
			*cEx, *cEy, *cEz;

	} CudaData;

    class FDTD {
    	private:
		    // Cuda data block (host copy)
		    CudaData _hcd;

 		    // number of Yee cells in X, Y, Z
		    int _Nx, _Ny, _Nz;

            // physical simulation size and Yee cell size in x,y,z
            double _X, _Y, _Z, _dx, _dy, _dz;

            // Wavelength defined in desired units
            double _wavelength;

            // spatial normalization factor
		    double _R;

            // PML parameters
            double _sigma, _alpha, _kappa, _pow;

		    // source array
            std::vector<SourceArray> _sources;

            // Complex array associated field at captured
            // time steps. Technically only one set of captured fields need to be
            // complex.
            complex128 *_Ex_t0, *_Ey_t0, *_Ez_t0,
                       *_Hx_t0, *_Hy_t0, *_Hz_t0,
                       *_Ex_t1, *_Ey_t1, *_Ez_t1,
                       *_Hx_t1, *_Hy_t1, *_Hz_t1;

		    // PML parameters
		    int _w_pml_x0, _w_pml_x1,
			    _w_pml_y0, _w_pml_y1,
			    _w_pml_z0, _w_pml_z1;

            /*!
             * Calculate the pml ramp function (which defines how the
             * PML values are scaled as you move towards the simulation
             * boundary)
             *
             * \param distance - the distance from the beginning of the PML.
             *                 0 = at pml boundary, 1 = at simulation bndry
             * \return The PML scaling factor.
             */
            double pml_ramp(double distance);

            /*!
             * Compute the PML parameters kappa, b, and c which depend on
             * distance from the PML boundary. Precomputing these values is
             * good from the standpoint of a) reducing duplicated code and b)
             * speeding things up. Computing the PML values actually appears
             * to be a major bottleneck in the computation.
             */
            void compute_pml_params();

        public:
            
		    FDTD();
  		    FDTD(int Nx, int Ny, int Nz);
            ~FDTD();

            /*!
             * Set the dimensions and discretization size of the simulation.
             *
             * Internally, we solve a non-dimensionalized form of Maxwell's Equations.
             * As a result, these values can be specified using any desired unit as
             * long as it is consistent with the wavelength. For example, if we want
             * to use a wavelength of 1.55 um, we might set wavelength=1.55 and then
             * express X=20.0, Y=5.0, Z=2.0 which would then correspond to a simulation
             * which is 20 um by 5 um by 2 um.
             *
             * \param X - The simulation width in the x direction.
             * \param Y - The simulation width in the y direction.
             * \param Z - The simulation width in the z direction.
             * \param dx - The simulation grid spacing in the x direction.
             * \param dy - The simulation grid spacing in the y direction.
             * \param dz - The simulation grid spacing in the z direction.
             */
            void set_physical_dims(double X, double Y, double Z,
                                         double dx, double dy, double dz);

            /*!
             * Set the wavelength of the simulation.
             *
             * Any unit for the wavelength may be chosen (um, cm, etc) so long
             * as all other length values are specified using the same unit.
             *
             * \param wavelength - The wavelength of the source excitation.
             */
            void set_wavelength(double wavelength);

            /*!
             * Set the time step used to update the fields.
             *
             * This time step must be set such that:
             *
             *      dt <= Sc * n * min([dx, dy, dz]) / c
             *
             * where c is the speed of light, n is the minimum refractive index in
             * the simulation, and Sc is the Courant number (=1/sqrt(3) in 3D).
             *
             * Note: this value needs to be non-dimensionalized (which requires multiplying
             * by \omega = 2*pi*c/wavelength). 
             *
             * \param dt - The time step.
             */
            void set_dt(double dt);

            /*!
             * Update the magnetic and electric fields num_times, starting at time start_time.
             *
             * \param start_time - The start time of the update = n*dt.
             * \param num_times -  the number of updates
             */
		    void update(double start_time, int num_times);

            /*!
             * Update the magnetic field at time t
             *
             * \param t - The time of the update = n*dt.
             */
            void update_H(double t);

            /*!
             * Update the electric field at time t
             *
             * \param t - The time of the update = n*dt+1/2*dt.
             */
            void update_E(double t);

            // PML configuration
            /*!
             * Set the PML widths along the simulation boundaries.
             *
             * \param xmin - The width of the PML at the minimum x boundary.
             * \param xmax - The width of the PML at the maximum x boundary.
             * \param ymin - The width of the PML at the minimum y boundary.
             * \param ymax - The width of the PML at the maximum y boundary.
             * \param zmin - The width of the PML at the minimum z boundary.
             * \param zmax - The width of the PML at the maximum z boundary.
             */
            void set_pml_widths(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

            /*!
             * Set the pml properties.
             *
             * The PML is defined using a conductivity sigma and two additional parameters
             * alpha and kappa. All three of these parameters are ramped from zero starting
             * at the PML boundary moving outwards towards the simulation boundary. The ramp
             * function is a polynomial of order pow.
             *
             * \param sigma - The PML sigma parameter (>= 1.0).
             * \param alpha - The PML alpha parameter (~0).
             * \param kappa - The PML kappa parameter (~1).
             * \param pow - The power of the ramp function f(s) = (s/smax)**pow
             */
            void set_pml_properties(double sigma, double alpha, double kappa, double pow);

            /*!
             * Build the PML arrays.
             *
             * This must be called after set_local_grid(...) and set_pml_widths(...)
             */
            void build_pml();

            /*!
             * Reset the PML arrays to zero.
             *
             * This must be called AFTER build_pml(...)
             */
            void reset_pml();

            // manage auxilary fields + amp/phase calculation

            /*!
             * Set preallocated global arrays which will store a snapshot of the field in time.
             *
             * These are used to calculate the amplitude and phase of the field.
             */
            void set_t0_arrays(complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                               complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0);

            /*!
             * Set preallocated global arrays which will store a snapshot of the field in time.
             *
             * These are used to calculate the amplitude and phase of the field.
             */
            void set_t1_arrays(complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
                               complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1);

            /*!
             * Record the field at the current time in the t0 arrays.
             */
            void capture_t0_fields();

            /*!
             * Record the field at the current time in the t1 arrays.
             */
            void capture_t1_fields();

            /*!
             * Calculate the complex amplitude and phase of the fields using
             * values for the fields at three different points in time.
             * 
             * In order for this to work, we need to first:
             *  1) Check that the fields have settled into sinusoidal behavior.
             *  2) Capture the fields at two previous times t0 and t1
             *     using capture_t0_fields and capture_t1_fields.
             *
             * \param t0 - The time at which the fields were recorded in the t0 array.
             * \param t1 - The time at which the fields were recorded in the t1 array.
             * \param t2 - The time of the most recent field update.
             */
            void calc_complex_fields_3T(double t0, double t1, double t2);

            // Manage source arrays
            /*!
             * Add a source which consists of a distribution of electric and magnetic
             * current density.
             *
             * The current densities may be complex-valued. Internally, a corresponding
             * amplitude and phase will be calculated.
             *
             * The current densities must be specified in a block of the simulation.
             * This block is defined by the lower index of the block and its size.
             *
             * Note: Magnetic current density is somewhat fictitious, but it allows us
             * to inject power in a single direction.
             *
             * \param Jx - The x component of the electric current density distribution.
             * \param Jy - The y component of the electric current density distribution.
             * \param Jz - The z component of the electric current density distribution.
             * \param Mx - The x component of the magnetic current density distribution.
             * \param My - The y component of the magnetic current density distribution.
             * \param Mz - The z component of the magnetic current density distribution.
             * \param i0 - The lower z index of the block.
             * \param j0 - The lower y index of the block.
             * \param k0 - The lower x index of the block.
             * \param I - The z width of the block.
             * \param J - The y width of the block.
             * \param K - The x width of the block.
             */
            void add_source(complex128 *Jx, complex128 *Jy, complex128 *Jz,
                            complex128 *Mx, complex128 *My, complex128 *Mz,
                            int i0, int j0, int k0, int I, int J, int K,
                            bool calc_phase);

            /*!
             * Clear the existing sources.
             */
            void clear_sources();

            /*!
             * Set the temporal properties of the source.
             *
             * The source is a ramped continuous wave excitation. This time dependence
             * is defined using a ramp time and a minimum source amplitude (which to 
             * some degree corresponds to a delay in the source).
             *
             * \param src_T - The source ramp time.
             * \param src_min - The minimum source amplitude.
             */
            void set_source_properties(double src_T, double src_min);

            /*!
             * The time dependence of the source.
             *
             * \param t - The current time.
             * \param phase - The phase of the sinusoid of the source.
             *
             * \return The time-dependent modulation value of the source.
             */
            double src_func_t(double t, double phase);

            /* Set the boundary conditions.
             *
             * The boundary conditions are defined using a 3 character string
             * which contains '0' (perfect electric conductor), 'E' (electric
             * field symmetry), 'H' (magnetic field symmetry), or 'P' (periodic).
             *
             * \param newbc - The boundary condition string.
             */
            void set_bc(char* newbc);
    };

};

extern "C" {
	    fdtd::FDTD* FDTD_new(int Nx, int Ny, int Nz);

        void FDTD_set_wavelength(fdtd::FDTD* fdtd, double wavelength);
        void FDTD_set_physical_dims(fdtd::FDTD* fdtd, 
                                    double X, double Y, double Z,
                                    double dx, double dy, double dz);
        void FDTD_set_dt(fdtd::FDTD* fdtd, double dt);
 	    void FDTD_update(fdtd::FDTD* fdtd, double start_time, int num_times);

        // Pml management
        void FDTD_set_pml_widths(fdtd::FDTD* fdtd, int xmin, int xmax,
                                                   int ymin, int ymax,
                                                   int zmin, int zmax);
        void FDTD_set_pml_properties(fdtd::FDTD* fdtd, double sigma, double alpha,
                                                       double kappa, double pow);
        void FDTD_build_pml(fdtd::FDTD* fdtd);
        void FDTD_reset_pml(fdtd::FDTD* fdtd);

        // auxillary array management
        void FDTD_set_t0_arrays(fdtd::FDTD* fdtd,
                                 complex128 *Ex_t0, complex128 *Ey_t0, complex128 *Ez_t0,
                                 complex128 *Hx_t0, complex128 *Hy_t0, complex128 *Hz_t0);
        void FDTD_set_t1_arrays(fdtd::FDTD* fdtd,
                                 complex128 *Ex_t1, complex128 *Ey_t1, complex128 *Ez_t1,
                                 complex128 *Hx_t1, complex128 *Hy_t1, complex128 *Hz_t1);

        double FDTD_calc_phase_3T(double t0, double t1, double t2, double f0, double f1, double f2);
        double FDTD_calc_amplitude_3T(double t0, double t1, double t2, double f0, double f1, double f2, double phase);

        void FDTD_capture_t0_fields(fdtd::FDTD* fdtd);
        void FDTD_capture_t1_fields(fdtd::FDTD* fdtd);
        void FDTD_calc_complex_fields_3T(fdtd::FDTD* fdtd, double t0, double t1, double t2);

        // Source management
        void FDTD_add_source(fdtd::FDTD* fdtd,
                             complex128 *Jx, complex128 *Jy, complex128 *Jz,
                             complex128 *Mx, complex128 *My, complex128 *Mz,
                             int i0, int j0, int k0, int I, int J, int K,
                             bool calc_phase);
        void FDTD_clear_sources(fdtd::FDTD* fdtd);

        void FDTD_set_source_properties(fdtd::FDTD* fdtd, double src_T, double src_min);

        double FDTD_src_func_t(fdtd::FDTD* fdtd, double t, double phase);

        // boundary conditions
        void FDTD_set_bc(fdtd::FDTD* fdtd, char* newbc);
};

#endif
