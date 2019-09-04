#include "CudaClipper.hpp"
#include <iostream>
#include <thrust/functional.h>

void* checkCudaMallocManaged(int nbytes)
{
	void *ptr;
	cudaError_t code = cudaMallocManaged((void **)&ptr, nbytes);
	if (code != cudaSuccess) {
		std::string err_string = std::string("CudaClipper managed memory allocation error:  ") +
			std::string(cudaGetErrorString(code));
		throw err_string;
	}
	return ptr;
}

class CudaPoint {
public:
	double x;
	double y;
	__device__ CudaPoint() :
		x(0), y(0) {};
	__device__ CudaPoint(double x, double y) :
		x(x), y(y) {};
	__host__ CudaPoint(BoostPoint pt) :
		x(pt.x()), y(pt.y()) {};
};

std::ostream& operator<< (std::ostream& os, const CudaPoint& pt)
{
	os << "" << pt.x << " " << pt.y;
	return os;
}

template <size_t MAX>
std::string wkt (std::array<CudaPoint, MAX> points, int N) {
	std::ostringstream oss;
	if (N) {
		oss << "POLYGON((";
		for (auto p = points.cbegin(); p < points.cbegin()+N; ++p)
			oss << p->x << " " << p->y << ",";
		oss << points.front().x << " " << points.front().y << "))";
	} else {
		oss << "POLYGON(())";
	}
	return oss.str();
}

enum inside_direction { CLIP_XGE, CLIP_YGE, CLIP_XLE, CLIP_YLE };
class CudaClipEdge {
private:
	double _clip_val;
	enum inside_direction _inside_dir;
public:
	__device__ CudaClipEdge (double clip_val, enum inside_direction inside_dir)
		: _clip_val(clip_val), _inside_dir(inside_dir) {
	};
	__device__ bool point_inside (const CudaPoint& p) {
		bool result;
		switch(_inside_dir) {
		case CLIP_XGE:
			result =  p.x >= _clip_val; break;
		case CLIP_YGE:
			result =  p.y >= _clip_val; break;
		case CLIP_XLE:
			result =  p.x <= _clip_val; break;
		case CLIP_YLE:
		default:
			result =  p.y <= _clip_val; break;
		}
		return result;
	};
	__device__ CudaPoint intersect (const CudaPoint& p0, const CudaPoint& p1) {
		CudaPoint p;
		switch(_inside_dir) {
		case CLIP_XGE:
		case CLIP_XLE:
			p.x = _clip_val;
			p.y = p0.y + (p1.y-p0.y)/(p1.x-p0.x)*(p.x-p0.x);
			break;
		case CLIP_YGE:
		case CLIP_YLE:
		default:
			p.y = _clip_val;
			p.x = p0.x + (p1.x-p0.x)/(p1.y-p0.y)*(p.y-p0.y);
		}
		return p;
	};
	friend std::ostream& operator<< (std::ostream& os, const CudaClipEdge& pt);
};

std::ostream& operator<< (std::ostream& os, const CudaClipEdge& edge)
{
	switch(edge._inside_dir) {
	case CLIP_XGE:
		os << "X >= "; break;
	case CLIP_YGE:
		os << "Y >= "; break;
	case CLIP_XLE:
		os << "X <= "; break;
	case CLIP_YLE:
		os << "Y <= "; break;
	default:
		os << "UNDEFINED ";
	}
	os << edge._clip_val;
	return os;
}

// from https://en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm
__global__ void cuda_trapezoid_box_intersection_area(double *cell_fractions,
													 CudaPoint point0, CudaPoint point1,
													 double j1off, double k1off,
													 double dx, double dy,
													 int Nx, int Ny)

{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const double cell_xmin=(k+k1off-0.5)*dx, cell_xmax=(k+k1off+0.5)*dx;
	const double cell_ymin=(j+j1off-0.5)*dy, cell_ymax=(j+j1off+0.5)*dy;
	const int index = j*Nx + k;
	const int IO_RING_MAX = 5;

	if (j<Ny && k<Nx) {
		// initialize output ring to the input trapezoid
		thrust::minimum<double> min;
		double trapezoid_xmin = min(cell_xmin, min(point0.x, point1.x)) - 1.0;
		CudaPoint output_ring[IO_RING_MAX] = {
			CudaPoint(trapezoid_xmin, point0.y),
			point0,
			point1,
			CudaPoint(trapezoid_xmin, point1.y) };
		int output_ring_N = 4;

		// initialize clip ring */
		CudaClipEdge clip_edges[4] = {
			CudaClipEdge(cell_xmin, CLIP_XGE),
			CudaClipEdge(cell_ymin, CLIP_YGE),
			CudaClipEdge(cell_xmax, CLIP_XLE),
			CudaClipEdge(cell_ymax, CLIP_YLE) };

		CudaPoint input_ring[IO_RING_MAX];
		int input_ring_N;

		for (auto clip_edge :  clip_edges) {
			for (int i=0; i<IO_RING_MAX; ++i) input_ring[i] = output_ring[i];
			input_ring_N = output_ring_N;
			output_ring_N = 0;

			for (int i = input_ring_N-1, j = 0;
				 j < input_ring_N;
				 i=j, j++)
			{		
				if (clip_edge.point_inside(input_ring[j])) { 
					if (!clip_edge.point_inside(input_ring[i])) {
						output_ring[output_ring_N++] = clip_edge.intersect(input_ring[i], input_ring[j]);
					}
					output_ring[output_ring_N++] = input_ring[j];
				} else {
					if (clip_edge.point_inside(input_ring[i])) {
						output_ring[output_ring_N++] = clip_edge.intersect(input_ring[i], input_ring[j]);
					}
				}
			}
		}

		// return area of output ring
		double intersection_area_2x = 0.0;
		for (int i = output_ring_N-1, j = 0;
			 j < output_ring_N;
			 i=j, j++)
		{
			intersection_area_2x += (output_ring[j].x + output_ring[i].x) * (output_ring[j].y - output_ring[i].y);
		}
		cell_fractions[index] -=  intersection_area_2x / (2*dx*dy);  // TODO: explain minus sign
	}
}

__global__ void zero_layer_values(thrust::complex<double>* layer_values, int Nx, int Ny)
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = j*Nx + k;
	if (j<Ny && k<Nx) {
		layer_values[index] = 0;
	}
}

__global__ void zero_cell_fractions(double* cell_fractions, int Nx, int Ny)
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = j*Nx + k;
	if (j<Ny && k<Nx) {
		cell_fractions[index] = 0;
	}
}

void CudaClipper::compute_ring_cell_fractions(BoostRing ring)
{
	zero_cell_fractions <<<numBlocks(), threadsPerBlock()>>> (_cell_fractions, Nx(), Ny());

	for (auto it=bg::segments_begin(ring); it!=bg::segments_end(ring); ++it) {
		CudaPoint point0=CudaPoint(*(it->first)), point1=CudaPoint(*(it->second));
		cuda_trapezoid_box_intersection_area <<<numBlocks(), threadsPerBlock() >>>
			(_cell_fractions,
			 point0, point1,
			 _j1 + _joff,
			 _k1 + _koff,
			 _dx, _dy, Nx(), Ny());
	}
}

__global__ void cuda_composite_cell_fraction(thrust::complex<double> *layer_values,
											 thrust::complex<double> matval,
											 double *cell_fractions,
											 int Nx, int Ny)
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = j*Nx + k;
	if (j<Ny && k<Nx) {
		layer_values[index] += matval * cell_fractions[index];
	}
}

void CudaClipper::composite_cell_fraction(thrust::complex<double> matval)
{
	cuda_composite_cell_fraction <<<numBlocks(), threadsPerBlock()>>>
		(_layer_values, matval, _cell_fractions, Nx(), Ny());
}

CudaClipper::CudaClipper(int k1, int k2, int j1, int j2, int i1, int i2,
			double koff, double joff,
			double dx, double dy) :
	_k1(k1), _k2(k2), _j1(j1), _j2(j2), _i1(i1), _i2(i2),
	_koff(koff), _joff(joff),
	_dx(dx), _dy(dy)
{
	_grid = (thrust::complex<double>*) checkCudaMallocManaged(sizeof(thrust::complex<double>) *
															  Nx() * Ny() * Nz());
	for (int i=0; i<Nx()*Ny()*Nz(); ++i)
		_grid[i] = 0.0;
	_layer_values = (thrust::complex<double>*) checkCudaMallocManaged(sizeof(thrust::complex<double>) *
																	  Nx() * Ny());
	_cell_fractions = (double*) checkCudaMallocManaged(sizeof(double) *
													   Nx() * Ny());
}

CudaClipper::~CudaClipper()
{
	cudaFree(_grid);
	cudaFree(_layer_values);
	cudaFree(_cell_fractions);
}


void CudaClipper::compute_layer_values(StructuredMaterialLayer* layer)
{
	zero_layer_values <<<numBlocks(), threadsPerBlock()>>> (_layer_values, Nx(), Ny());
	for (auto polymat : layer->get_polymats()) {
		for (auto bpoly : polymat->get_bpolys()) {
			auto outer_ring = bpoly.outer();
			compute_ring_cell_fractions(outer_ring);
			composite_cell_fraction(polymat->get_matval());
			for (auto inner_ring : bpoly.inners()) {
				compute_ring_cell_fractions(inner_ring);
				composite_cell_fraction(polymat->get_matval());
			}
		}
	}
}

__global__ void cuda_composite_layer(thrust::complex<double>* slice,
									 thrust::complex<double>* layer_values, double alpha,
									 int Nx, int Ny)
{
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = j*Nx + k;
	if (j<Ny && k<Nx) {
		slice[index] += alpha * layer_values[index];
	}
}

void CudaClipper::composite_layer_values_into_slice(double alpha, int z_index)
{
	cuda_composite_layer <<<numBlocks(), threadsPerBlock()>>>
		(&_grid[(z_index-_i1)*Nx()*Ny()], _layer_values, alpha, Nx(), Ny());
}

void CudaClipper::return_grid_values(std::complex<double> *grid)
{
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) {
		printf("CudaClipper sync kernel error: %s\n", cudaGetErrorString(errSync));
		exit(-1);
	}
	if (errAsync != cudaSuccess) {
		printf("CudaClipper async kernel error: %s\n", cudaGetErrorString(errAsync));
		exit(-1);
	}
	for (int i=0; i<Nx()*Ny()*Nz(); ++i) {
		grid[i].real(_grid[i].real());
		grid[i].imag(_grid[i].imag());
	}
}
