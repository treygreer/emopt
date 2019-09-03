#include "CudaClipper.hpp"
#include <iostream>

class CudaPoint {
public:
	double x;
	double y;
	CudaPoint() :
		x(0), y(0) {};
	CudaPoint(double x, double y) :
		x(x), y(y) {};
	CudaPoint(BoostPoint pt) :
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
	CudaClipEdge (double clip_val, enum inside_direction inside_dir)
		: _clip_val(clip_val), _inside_dir(inside_dir) {
	};
	bool point_inside (const CudaPoint& p) {
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
	CudaPoint intersect (const CudaPoint& p0, const CudaPoint& p1) {
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
double trapezoid_box_intersection_area(CudaPoint& point0, CudaPoint& point1,
									   double cell_xmin, double cell_xmax, double cell_ymin, double cell_ymax)
{
	const int IO_RING_MAX = 5;

	// initialize output ring to the input trapezoid
	double trapezoid_xmin = std::min(cell_xmin, std::min(point0.x, point1.x)) - 1.0;
	std::array<CudaPoint, IO_RING_MAX> output_ring = { CudaPoint(trapezoid_xmin, point0.y),
													   point0,
													   point1,
													   CudaPoint(trapezoid_xmin, point1.y) };
	int output_ring_N = 4;

	// initialize clip ring */
	const std::array<CudaClipEdge, 4> clip_edges = { CudaClipEdge(cell_xmin, CLIP_XGE),
													 CudaClipEdge(cell_ymin, CLIP_YGE),
													 CudaClipEdge(cell_xmax, CLIP_XLE),
													 CudaClipEdge(cell_ymax, CLIP_YLE) };

	std::array<CudaPoint, IO_RING_MAX> input_ring; int input_ring_N;

	//std::cerr << "*** intersect ********\n";
	//std::cerr << "   point0 = " << point0 << " point1=" << point1 << "\n";

	for (auto clip_edge :  clip_edges) {
		input_ring = output_ring;
		input_ring_N = output_ring_N;
		output_ring_N = 0;
		//std::cerr << "    clip edge: " << clip_edge << "\n";
		//std::cerr << "      input ring = " << wkt(input_ring, input_ring_N) << "\n";

		for (auto input_pt0 = input_ring.cbegin()+input_ring_N-1, input_pt1 = input_ring.cbegin();
			 input_pt1 < input_ring.cbegin()+input_ring_N;
			 input_pt0 = input_pt1, input_pt1++)
		{		
			//std::cerr << "        input_pt0 = " << *input_pt0 << " input_pt1=" << *input_pt1;
			if (clip_edge.point_inside(*input_pt1)) { 
				//std::cerr << " pt1_in";
				if (!clip_edge.point_inside(*input_pt0)) {
					//std::cerr << " pt0_out(isect=" << clip_edge.intersect(*input_pt0, *input_pt1) << ") ";
					output_ring[output_ring_N++] = clip_edge.intersect(*input_pt0, *input_pt1);
				}
				output_ring[output_ring_N++] = *input_pt1;
			} else {
				//std::cerr << "  pt1_out";
				if (clip_edge.point_inside(*input_pt0)) {
					//std::cerr << " pt0_in(isect=" << clip_edge.intersect(*input_pt0, *input_pt1) << ") ";
					output_ring[output_ring_N++] = clip_edge.intersect(*input_pt0, *input_pt1);
				}
			}
			//std::cerr << "\n";
		}
		//std::cerr << "      output ring = " << wkt(output_ring, output_ring_N) << "\n";
	}

	// return area of output ring
	double output_area_2x = 0.0;
	for (auto output_pt0 = output_ring.cbegin()+output_ring_N-1, output_pt1 = output_ring.cbegin();
		 output_pt1 < output_ring.cbegin()+output_ring_N;
		 output_pt0 = output_pt1, output_pt1++)
	{
		output_area_2x += (output_pt1->x + output_pt0->x) * (output_pt1->y - output_pt0->y);
	}
	//std::cerr << "    area = " << output_area_2x * 0.5 << "\n";
	return output_area_2x * 0.5;
}


void CudaClipper::zero_layer_values()
{
	const int Nx = _k2-_k1;
	const int Ny = _k2-_k1;
	for(int i = 0; i<Nx*Ny; ++i) 
		_layer_values[i] = 0;
}

void CudaClipper::compute_ring_cell_fractions(BoostRing ring)
{
	const int Nx = _k2-_k1;
	const int Ny = _k2-_k1;
	for(int i = 0; i<Nx*Ny; ++i) 
		_cell_fractions[i] = 0;

	for (auto it=bg::segments_begin(ring); it!=bg::segments_end(ring); ++it) {
		CudaPoint point0=CudaPoint(*(it->first)), point1=CudaPoint(*(it->second));
		for(int j = _j1; j < _j2; j++) {
			for(int k = _k1; k < _k2; k++) {
				int index = (j-_j1)*Ny+k-_k1;
				const double cell_xmin=(k+_koff-0.5)*_dx, cell_xmax=(k+_koff+0.5)*_dx;
				const double cell_ymin=(j+_joff-0.5)*_dy, cell_ymax=(j+_joff+0.5)*_dy;
				double intersection_area = trapezoid_box_intersection_area(point0, point1,
																		   cell_xmin, cell_xmax,
																		   cell_ymin, cell_ymax);
				intersection_area = -intersection_area;  // TODO: explain this
				_cell_fractions[index] +=  intersection_area / (_dx*_dy);
			}
		}
	}
}
			
void CudaClipper::composite_cell_fraction(std::complex<double> matval)
{
	const int Ny = _k2 - _k1;
	for(int j = _j1; j < _j2; j++) {
		for(int k = _k1; k < _k2; k++) {
			int index = (j-_j1)*Ny+k-_k1;
			_layer_values[index] += matval * _cell_fractions[index];
		}
	}
};

CudaClipper::CudaClipper(int k1, int k2, int j1, int j2, int i1, int i2,
			double koff, double joff,
			double dx, double dy) :
	_k1(k1), _k2(k2), _j1(j1), _j2(j2), _i1(i1), _i2(i2),
	_koff(koff), _joff(joff),
	_dx(dx), _dy(dy)
{
	const int Nx = k2-k1;
	const int Ny = j2-j1;
	const int Nz = i2-i1;
	_grid = new std::complex<double>[Nx * Ny * Nz];
	for (int i=0; i<Nx*Ny*Nz; ++i)
		_grid[i] = 0.0;
	_layer_values = new std::complex<double>[Ny * Nx];
	_cell_fractions = new double[Ny * Nx];
}

CudaClipper::~CudaClipper()
{
	delete[] _grid;
	delete[] _layer_values;
	delete[] _cell_fractions;
}


void CudaClipper::compute_layer_values(StructuredMaterialLayer* layer)
{
	zero_layer_values();

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

void CudaClipper::composite_layer_values_into_slice(double alpha, int z_index)
{
	const int Nx = _k2-_k1;
	const int Ny = _j2-_j1;
	for(int j = _j1; j < _j2; j++) {
		for(int k = _k1; k < _k2; k++) {
			int layer_index = (j-_j1)*Nx + (k-_k1);
			int grid_index = layer_index + (z_index-_i1)*Nx*Ny;
			_grid[grid_index] += alpha * _layer_values[layer_index];
		}
	}
}

void CudaClipper::return_grid_values(std::complex<double> *grid)
{
	const int Nx = _k2-_k1;
	const int Ny = _j2-_j1;
	const int Nz = _i2-_i1;
	std::memcpy(grid, _grid, Nx*Ny*Nz*sizeof(std::complex<double>));
}
