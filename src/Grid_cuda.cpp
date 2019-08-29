#include "Grid_cuda.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace GridCuda;

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


/**************************************** Materials ****************************************/

//------------------------------ PolyMat ------------------------------------/

PolyMat::PolyMat(double* x, double* y, int n, std::complex<double> matval) :
	_matval(matval)
{
	_bpolys.resize(1);
    for(int i = 0; i < n; i++) {
        bg::append(_bpolys, bg::make<BoostPoint>(x[i], y[i]));
    }

    // correct the geometry
    bg::correct(_bpolys);
}

PolyMat::PolyMat(BoostMultiPolygon bpolys, std::complex<double> matval) :
	_matval(matval), _bpolys(bpolys)
{
}

PolyMat::PolyMat(PolyMat *pm) :
	_matval(pm->_matval), _bpolys(pm->_bpolys)
{
}

PolyMat::~PolyMat()
{
	_bpolys.clear();
}

void PolyMat::clip(BoostBox box)
{
	BoostMultiPolygon clipped_bpolys;
	bg::intersection(_bpolys, box, clipped_bpolys);
	_bpolys.clear();
	_bpolys = clipped_bpolys;
}

void PolyMat::subtract(BoostMultiPolygon bpolys)
{
	BoostMultiPolygon diff_bpolys;
	bg::difference(_bpolys, bpolys, diff_bpolys);
	_bpolys.clear();
	_bpolys = diff_bpolys;
}

/////////////////////////////////////////////////////////////////////////////////////
// StructuredMaterialLayer
/////////////////////////////////////////////////////////////////////////////////////
StructuredMaterialLayer::StructuredMaterialLayer(double X, double Y, double dx, double dy,
												 double background, double z_base) :
	_X(X), _Y(Y), _dx(dx), _dy(dy), _background(background), _z_base(z_base)
{
	// start with background material, value = 1.0
	double background_xs[5] = {-dx, X+dx, X+dx, -dx};
	double background_ys[5] = {-dy, -dy,  Y+dy, Y+dy};

	std::complex<double> background_material(1.0, 0.0);
	PolyMat *background_polymat = new PolyMat(background_xs, background_ys, 4, background_material);

	// make material bounding box (useful for area assertions)
    bg::envelope(background_polymat->get_bpolys(), _envelope);

    _polymats.push_front(background_polymat);
}

StructuredMaterialLayer::~StructuredMaterialLayer() {
	for (auto pm : _polymats) {
		delete pm;
	}
}

/* Allocate and add polymat to this structured material layer.
 */
void StructuredMaterialLayer::add_polymat(PolyMat* polymat)
{
	PolyMat *new_polymat = new PolyMat(polymat);

	// clip polymat to material envelope (useful for area assertions)
	new_polymat->clip(_envelope);
	
	// subtract polymat from preceeding polymats
	for (auto pm = _polymats.begin(); pm != _polymats.end(); ) {
		(*pm)->subtract(new_polymat->get_bpolys());
		// remove pm if it's been clipped completely away
		if ((*pm)->is_empty()) {
			delete *pm;
			pm = _polymats.erase(pm);
		} else {
			++pm;
		}
	}

	// add new polymat to list
	_polymats.push_back(new_polymat);

	this->verify_area();
}

void StructuredMaterialLayer::verify_area()
{
	double total_area = 0.0;
	_polys_valid = true;
	for (auto polymat : _polymats) {
		const BoostMultiPolygon bpolys = polymat->get_bpolys();
		if (!bg::is_valid(bpolys)) {
			_polys_valid = false;
			std::cerr << "WARNING: multi_polygon not valid: " << bg::wkt(bpolys) << "\n";
		}
		double poly_area = bg::area(polymat->get_bpolys());
		if (poly_area < 0.0) {
			std::cerr << "ERROR: poly_area less than zero: " << poly_area << "\n";
			exit(-1);
		}
		total_area += poly_area;
	}
	double envelope_area = bg::area(_envelope);
	if (_polys_valid && (fabs(total_area - envelope_area) / envelope_area) > 1e-12) {
		std::cerr << "ERROR: total_area " << total_area << " != envelope_area " << envelope_area << "\n";
		exit(-1);
	}
}

void StructuredMaterialLayer::add_polymats(std::list<PolyMat*> polymats)
{
    std::list<PolyMat*>::iterator it;
    for (it = polymats.begin(); it != polymats.end(); it++) {
        add_polymat(*it);
    }
}

////////////////////////////////////////////////////////////////////////////////////
// ConstantMaterial3D
////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial3D::ConstantMaterial3D(std::complex<double> value)
{
    _value = value;
}

void ConstantMaterial3D::get_values(std::complex<double>* grid, int k1, int k2, int j1, int j2,
                                    int i1, int i2,
									double koff, double joff, double ioff)
{
    int N = k2 - k1,
        M = j2 - j1;

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                grid[(i-i1)*N*M + (j-j1)*N + k-k1] = _value;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// StructuredMaterial3D
////////////////////////////////////////////////////////////////////////////////////

StructuredMaterial3D::StructuredMaterial3D(double X, double Y, double Z,
                                           double dx, double dy, double dz,
	                                       double background) :
                                           _X(X), _Y(Y), _Z(Z), 
                                           _dx(dx), _dy(dy), _dz(dz),
										   _background(background)
{
	// add empty StructuredMaterialLayers at z=_BOTTOM_Z and z=_TOP_Z
	_layers.push_back(new StructuredMaterialLayer(_X, _Y, _dx, _dy, _background, _MIN_Z));
	_layers.push_back(new StructuredMaterialLayer(_X, _Y, _dx, _dy, _background, _MAX_Z));
}

// We allocate memory -- Need to free it!
StructuredMaterial3D::~StructuredMaterial3D()
{
	for(auto layer : _layers) {
        delete layer;
    }
}

void StructuredMaterial3D::add_polymat(PolyMat* polymat, double z1, double z2)
{
    // Make sure the layer has a thickness
    if(z1 == z2) {
        std::cerr << "Warning in Structured3DMaterial: Provided layer has no \
                      thickness. It will be ignored." << std :: endl;

        return;
    }
    else if(z2 < z1) {
        std::cerr << "Warning in Structured3DMaterial: Provided layer has negative \
                      thickness. It will be ignored." << std :: endl;

        return;
    }

    // now we insert the beginning and end point of the layer one at a time, breaking
    // up or inserting new layers as necessary
    for(const double point_z : std::array<double,2> {z1, z2}) {
		assert(point_z > _MIN_Z && point_z < _MAX_Z);

		// figure out where the point is going to go
		auto layer_before = _layers.begin();
		auto layer_after = _layers.begin()++;
		while (layer_after != _layers.end() &&
			   point_z > (*layer_after)->z_base())
		{
			layer_before = layer_after++;
		}
		assert(point_z > (*layer_before)->z_base());
		assert(point_z <= (*layer_after)->z_base());

		if(point_z != (*layer_after)->z_base()) {  // if the point to insert is not already in the stack 
			StructuredMaterialLayer* layer = new StructuredMaterialLayer(_X, _Y, _dx, _dy, _background, point_z);
			layer->add_polymats( (*layer_before)->get_polymats() );
			_layers.insert(layer_after, layer);  // insert before layer_after
		}
    }

    // Finally, insert the supplied PolyMat into the desired locations
    for (auto itl : _layers) {
        double z = itl->z_base();
        if(z >= z1 && z < z2) {
            itl->add_polymat(polymat);
        }
    }

	/*
	std::cout << "SM3D::add_polymat results:\n";
    for(auto layer : _layers) {
		std::cout << "   layer " << layer << " at z=" << layer->z_base() << "...\n";
        std::list<PolyMat*> polymats = layer->get_polymats();

        for(auto pm : layer->get_polymats()) {
            std::cout << "       " << pm << "  area=" << pm->get_area() << " mat=" << pm->get_matval().real() << "\n";
			std::cout << "       " << bg::wkt(pm->get_bpolys()) << "\n";
		}
    }
	*/

    // aaannnddd we're done!
}

class CudaGrid {
private:
	std::vector<std::complex<double>> _grid;
	std::vector<std::complex<double>> _layer_values;
	std::vector<double> _cell_fractions;
	int _k1, _k2, _j1, _j2, _i1, _i2;
	double _koff, _joff;
	double _dx, _dy;


	void zero_layer_values()
		{
			const int Ny = _k2 - _k1;
			for(int j = _j1; j < _j2; j++) 
				for(int k = _k1; k < _k2; k++) 
					_layer_values[(j-_j1)*Ny+k-_k1] = 0;
		}

	void compute_ring_cell_fractions(BoostRing ring)
		{
			const int Ny = _k2 - _k1;
			for(int j = _j1; j < _j2; j++) {
				for(int k = _k1; k < _k2; k++) {
					int index = (j-_j1)*Ny+k-_k1;
					_cell_fractions[index] = 0.0;
				}
			}

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
						if (y0 < y1) intersection_area = -intersection_area;
						_cell_fractions[index] +=  intersection_area / (_dx*_dy);
					}
				}
			}
		}
			
	void composite_cell_fraction(std::complex<double> matval)
		{
			const int Ny = _k2 - _k1;
			for(int j = _j1; j < _j2; j++) {
				for(int k = _k1; k < _k2; k++) {
					int index = (j-_j1)*Ny+k-_k1;
					_layer_values[index] += matval * _cell_fractions[index];
				}
			}
		};

public:
	CudaGrid(int k1, int k2, int j1, int j2, int i1, int i2,
			 double koff, double joff,
		     double dx, double dy) :
		_k1(k1), _k2(k2), _j1(j1), _j2(j2), _i1(i1), _i2(i2),
		_koff(koff), _joff(joff),
		_dx(dx), _dy(dy)
		{
			const int Nx = k2-k1;
			const int Ny = j2-j1;
			const int Nz = i2-i1;
			_grid.resize(Nz*Ny*Nx, 0.0);
			_layer_values.resize(Ny*Nx, 0.0);
			_cell_fractions.resize(Ny*Nx, 0.0);
		};

	void compute_layer_values(StructuredMaterialLayer* layer)
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
		};

	void composite_layer_values_into_slice(double alpha, int z_index)
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
		};

	void return_grid_values(std::complex<double> *grid)
		{
			const int Nx = _k2-_k1;
			const int Ny = _j2-_j1;
			const int Nz = _i2-_i1;
			std::memcpy(grid, _grid.data(), Nx*Ny*Nz*sizeof(std::complex<double>));
		};
};

// Note that this takes a 1D array!
void StructuredMaterial3D::get_values(std::complex<double>* grid,
									  int k1, int k2, 
									  int j1, int j2, 
									  int i1, int i2, 
									  double koff, double joff, double ioff)
{
	auto layer = _layers.begin();
	auto layer_next = _layers.begin()++;
	CudaGrid cuda_grid(k1, k2, j1, j2, i1, i2, koff, joff, _dx, _dy);
	StructuredMaterialLayer* values_layer = NULL;

    for(int slice_idx = i1; slice_idx < i2; slice_idx++) {  // z index

		double       slice_z_min = (slice_idx+ioff-0.5) * _dz;
		const double slice_z_max = (slice_idx+ioff+0.5) * _dz;

		while (layer_next != _layers.end()) {
			double layer_z_base = (*layer)->z_base();
			double layer_z_top = (*layer_next)->z_base();

			if (slice_z_min >= layer_z_base) {
				if (values_layer != *layer) {
					cuda_grid.compute_layer_values(*layer);
					values_layer = *layer;
				}
				if (slice_z_max <= layer_z_top) {
					cuda_grid.composite_layer_values_into_slice((slice_z_max - slice_z_min) / _dz,
																slice_idx);
					break;  // break out of layer loop: stay on this layer and go to next slice
				}
				else if (slice_z_min < layer_z_top) {
					assert(slice_z_max > layer_z_top);
					cuda_grid.composite_layer_values_into_slice((layer_z_top - slice_z_min) / _dz,
																slice_idx);
					slice_z_min = layer_z_top;  // go to next layer and stay on this slice
				}
				else {
					assert(slice_z_min >= layer_z_top);  // go to next layer and stay on this slice
				}
            } // if slice_z_min >= layer_z_base
			layer = layer_next++;
        } // layer loop
    } // slice loop

	cuda_grid.return_grid_values(grid);
}
