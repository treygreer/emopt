#include "Grid_cuda.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace GridCuda;


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

	BoostBox bbox;
    bg::envelope(_bpolys, bbox);
	std::cout << "new polygon "<< this << "... matval=" << matval.real() <<
		", bbox=" << bg::wkt(bbox) <<
		", area=" << bg::area(_bpolys) << "\n";
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
// StructuredMaterial2D
/////////////////////////////////////////////////////////////////////////////////////
StructuredMaterial2D::StructuredMaterial2D(double X, double Y, double dx, double dy) :
	_X(X), _Y(Y), _dx(dx), _dy(dy)
{
	// start with background material, value = 1.0
	double background_xs[5] = {-dx, X+dx, X+dx, -dx};
	double background_ys[5] = {-dy, -dy,  Y+dy, Y+dy};

	std::complex<double> background_material(1.0, 0.0);
	PolyMat *background_polymat = new PolyMat(background_xs, background_ys, 4, background_material);

	// make material bounding box (useful for area assertions)
    bg::envelope(background_polymat->get_bpolys(), _envelope);

    _polymats.push_front(background_polymat);
	std::cout << "SM2D::new  area=" << bg::area(_envelope) << " background_polymat=" << background_polymat << "\n";
}

StructuredMaterial2D::~StructuredMaterial2D() {
	for (auto pm : _polymats) {
		delete pm;
	}
}

/* Allocate and add polymat to this 2D structured material.
 */
void StructuredMaterial2D::add_polymat(PolyMat* polymat)
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

void StructuredMaterial2D::verify_area()
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

void StructuredMaterial2D::add_polymats(std::list<PolyMat*> polymats)
{
    std::list<PolyMat*>::iterator it;
    for (it = polymats.begin(); it != polymats.end(); it++) {
        add_polymat(*it);
    }
}

typedef struct {
	double x;
	double y;
} CudaPoint;

std::ostream& operator<< (std::ostream& os, const CudaPoint& pt)
{
	os << "(" << pt.x << ", " << pt.y << ")";
	return os;
}

typedef struct {
	int rng_sign;
	int rng_num_points;
	int rng_point0_idx;
} CudaRing;

typedef struct {
	std::complex<double> cpm_matval;
	int cpm_num_rings;
	int cpm_ring0_idx;
} CudaPolyMat;

const int CUDA_MAX_POLYMATS = 20;
const int CUDA_MAX_RINGS = 200;
const int CUDA_MAX_POINTS = 16*1024;

//////////////////// FIXME TODO:  BOUNDS CHECKING!!!!!!!!!!!!!!!!!

CudaPolyMat cuda_polymats[CUDA_MAX_POLYMATS];
CudaRing cuda_rings[CUDA_MAX_RINGS];
CudaPoint cuda_points[CUDA_MAX_POINTS];
int cuda_total_polymats = 0;
int cuda_total_rings = 0;
int cuda_total_points = 0;

std::ostream& operator<< (std::ostream& os, const CudaRing& ring)
{
	os << "LINESTRING( ";
	for (int i=ring.rng_point0_idx; i<ring.rng_point0_idx+ring.rng_num_points; ++i) {
		os << cuda_points[i] << ", ";
	}
	os << cuda_points[ring.rng_point0_idx] << ")";
	
	return os;
}

void StructuredMaterial2D::finalize()
{
	_cuda_polymat0_idx = cuda_total_polymats;
	for (auto polymat : _polymats) {
		CudaPolyMat* cuda_polymat = &cuda_polymats[cuda_total_polymats++];
		cuda_polymat->cpm_matval = polymat->get_matval();
		cuda_polymat->cpm_ring0_idx = cuda_total_rings;
		for (auto bpoly : polymat->get_bpolys()) {
			/* outer ring */
			BoostRing boost_ring = bpoly.outer();
			CudaRing* cuda_ring = &cuda_rings[cuda_total_rings++];
			cuda_ring->rng_sign = +1;  // outer ring
			cuda_ring->rng_point0_idx = cuda_total_points;
			for (auto boost_point = bg::points_begin(boost_ring);
				 boost_point != bg::points_end(boost_ring);
				 ++boost_point)
			{
				CudaPoint* cuda_point = &cuda_points[cuda_total_points++];
				cuda_point->x = boost_point->x();
				cuda_point->y = boost_point->y();
			}
			cuda_ring->rng_num_points = cuda_total_points - cuda_ring->rng_point0_idx;

			/* inner rings */
			for (auto boost_ring : bpoly.inners()) {
				CudaRing* cuda_ring = &cuda_rings[cuda_total_rings++];
				cuda_ring->rng_sign = -1;  // inner ring
				cuda_ring->rng_point0_idx = cuda_total_points;
				for (auto boost_point = bg::points_begin(boost_ring);
					 boost_point != bg::points_end(boost_ring);
					 ++boost_point)
				{
					CudaPoint* cuda_point = &cuda_points[cuda_total_points++];
					cuda_point->x = boost_point->x();
					cuda_point->y = boost_point->y();
				}
				cuda_ring->rng_num_points = cuda_total_points - cuda_ring->rng_point0_idx;
			}
		}
		cuda_polymat->cpm_num_rings = cuda_total_rings - cuda_polymat->cpm_ring0_idx;
	}
	_cuda_num_polymats = cuda_total_polymats - _cuda_polymat0_idx;
}

void StructuredMaterial2D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, double sx, double sy)
{
    int N = k2 - k1;
    for(int j = j1; j < j2; j++) {
        for(int k = k1; k < k2; k++) {
            grid((j-j1)*N+k-k1) = get_value(k+sx, j+sy);
        }
    }
}

class Line {
private:
	double _a;
	double _b;
	double _c;
public:
	Line(CudaPoint* p1, CudaPoint* p2) {
		_a = p2->y - p1->y;
		_b = p1->x - p2->x;
		_c = p2->x*p1->y - p1->x*p2->y;
	}

	bool point_inside(CudaPoint* p) {
        return _a*p->x + _b*p->y + _c >= 0.0;
	}

	bool intersect(const Line l, CudaPoint* p) {
		double ap = _b*l._c - l._b*_c;
		double bp = l._a*_c - _a*l._c;
		double cp = _a*l._b - l._a*_b;
		if (cp != 0) {
			p->x = ap/cp;
			p->y = bp/cp;
			return true;
		} else {
			p->x = NAN;
			p->y = NAN;
			return false;
		}
	}
};

// from https://en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm
double StructuredMaterial2D::trapezoid_box_intersection_area(int pt0_idx, int pt1_idx,
															 double x_min, double x_max, double y_min, double y_max)
{
	const int IO_RING_N = 5;

	// initialize output ring to the input trapezoid
	double x0 = cuda_points[pt0_idx].x;
	double y0 = cuda_points[pt0_idx].y;
	double x1 = cuda_points[pt1_idx].x;
	double y1 = cuda_points[pt1_idx].y;
	double trapezoid_xmin = std::min(x_min, std::min(x0, x1)) - _dx;
	CudaPoint output_ring[IO_RING_N] = { CudaPoint({trapezoid_xmin, y0}),
										 cuda_points[pt0_idx],
										 cuda_points[pt1_idx],
										 CudaPoint({trapezoid_xmin, y1}) };
	int output_ring_N = 4;

	// initialize clip ring to {x,y} {min,max} box */
	const int clip_ring_N = 4;
	CudaPoint clip_ring[clip_ring_N] = { CudaPoint({x_min, y_min}),
										 CudaPoint({x_max, y_min}),
										 CudaPoint({x_max, y_max}),
										 CudaPoint({x_min, y_max}) };

	std::cerr << "*** intersect ********\n";
	std::cerr << "   output ring = " << &output_ring << "\n";
	std::cerr << "   clip ring = " << &clip_ring << "\n";

	CudaPoint input_ring[IO_RING_N]; int input_ring_N;

	for (auto clip_pt0 = &clip_ring[clip_ring_N-1], clip_pt1 = &clip_ring[0];
		 clip_pt1 < &clip_ring[clip_ring_N];
		 clip_pt0 = clip_pt1, clip_pt1++)
	{
		Line clip_line(clip_pt0, clip_pt1);
		// copy output ring to input ring and clear output ring
		for (int i=0; i<IO_RING_N; ++i) input_ring[i] = output_ring[i];
		input_ring_N = output_ring_N;
		output_ring_N = 0;

		for (auto input_pt0 = &input_ring[input_ring_N-1], input_pt1 = &input_ring[0];
			 input_pt1 < &input_ring[input_ring_N];
			 input_pt0 = input_pt1, input_pt1++)
		{		
			Line input_line(input_pt0, input_pt1);
			CudaPoint intersecting_point;
			clip_line.intersect(input_line, &intersecting_point);

			if (clip_line.point_inside(input_pt1)) { 
				if (!clip_line.point_inside(input_pt0)) {
					output_ring[output_ring_N++] = intersecting_point;
				}
				output_ring[output_ring_N++] = *input_pt1;
			} else {
				if (clip_line.point_inside(input_pt0)) {
					output_ring[output_ring_N++] = intersecting_point;
				}
			} 
		}
	}
	std::cerr << "   output ring = " << output_ring << "\n";

	// return area of output ring
	double output_area_2x = 0.0;
	for (auto output_pt0 = &output_ring[output_ring_N-1], output_pt1 = &output_ring[0];
			 output_pt1 < &output_ring[output_ring_N];
			 output_pt0 = output_pt1, output_pt1++)
	{
		output_area_2x += (output_pt1->x + output_pt0->x) * (output_pt1->y - output_pt0->y);
	}

	std::cerr << "    area = " << output_area_2x * 0.5 << "\n";
	return output_area_2x * 0.5;
}

double StructuredMaterial2D::ring_cell_intersection_area(int ring_idx,
														 double x_min, double x_max,
														 double y_min, double y_max,
														 bool debug)
{
	CudaRing* cuda_ring = &cuda_rings[ring_idx];
	int ring_num_points = cuda_ring->rng_num_points;
	int ring_point0_idx = cuda_ring->rng_point0_idx;
	double trapezoid_area_sum = 0.0;

	//BoostBox cell_bbox = BoostBox(BoostPoint(x_min, y_min), BoostPoint(x_max, y_max));

	for (int pt0_idx = ring_point0_idx + ring_num_points - 1, pt1_idx = ring_point0_idx;
		 pt1_idx - ring_point0_idx < ring_num_points;
		 pt0_idx = pt1_idx, pt1_idx++)
	{
		/*
		CudaPoint* pt0 = &cuda_points[pt0_idx];
		CudaPoint* pt1 = &cuda_points[pt1_idx];
		BoostPoint bp0=BoostPoint(pt0->x, pt0->y);
		BoostPoint bp1=BoostPoint(pt1->x, pt1->y);
		double x0=bg::get<0>(bp0);
		double x1=bg::get<0>(bp1);
		double y0=bg::get<1>(bp0);
		double y1=bg::get<1>(bp1);
		double trapezoid_xmin = std::min(x_min, std::min(x0, x1)) - _dx;
		BoostRing trapezoid { {trapezoid_xmin, y0}, bp0, bp1, {trapezoid_xmin, y1} };
		bg::correct(trapezoid);
		double intersection_area = trapezoid_box_intersection_area(trapezoid, cell_bbox);
		if (y0 < y1) intersection_area = -intersection_area;
		*/

		trapezoid_area_sum += trapezoid_box_intersection_area(
			pt0_idx, pt1_idx, x_min, x_max, y_min, y_max);
	}
	return trapezoid_area_sum;

	/*
	BoostMultiPolygon intersection_bpolys;
	BoostRing boost_ring;
	for (int pt_idx=rng_point0_idx; pt_idx-rng_point0_idx<rng_num_points; ++pt_idx) {
		CudaPoint* cuda_point = &cuda_points[pt_idx];
		bg::append(boost_ring, BoostPoint(cuda_point->x, cuda_point->y));
	}
	bg::correct(boost_ring);
		
	bg::intersection(boost_ring, cell_bbox, intersection_bpolys);
	double boost_area = bg::area(intersection_bpolys);

	if (fabs(boost_area-trapezoid_area_sum) / boost_area > 1e-12 &&
		fabs(boost_area-trapezoid_area_sum) > 1e-12) 
	{
	    std::cerr << "boost_area=" << boost_area << "\n";
		std::cerr << "  ring: " << bg::wkt(boost_ring) << "\n";
		std::cerr << "  box:  " << bg::wkt(cell_bbox) << "\n";
		std::cerr << "  intersection:  " << bg::wkt(intersection_bpolys) << "\n";
		//exit(-1);
		}
	return boost_area;
	*/
}

std::complex<double> StructuredMaterial2D::get_value(double cell_k, double cell_j)
{
	std::complex<double> cell_value = 0.0;
	double inv_cell_area = 1.0 / (_dx*_dy);
	double fraction_sum = 0.0;
	double x_min = _dx*(cell_k-0.5), x_max = _dx*(cell_k+0.5);
	double y_min = _dy*(cell_j-0.5), y_max = _dy*(cell_j+0.5);
	for (int pm_idx=_cuda_polymat0_idx; pm_idx<_cuda_polymat0_idx+_cuda_num_polymats; ++pm_idx) {
		CudaPolyMat* cuda_polymat = &cuda_polymats[pm_idx];
		std::complex<double> material_value = cuda_polymat->cpm_matval;
		int ring0_idx = cuda_polymat->cpm_ring0_idx;
		int num_rings = cuda_polymat->cpm_num_rings;
		for (int ring_idx=ring0_idx; ring_idx-ring0_idx < num_rings; ++ring_idx) {
			int ring_sign = cuda_rings[ring_idx].rng_sign;
			double fraction = ring_sign * inv_cell_area *
				ring_cell_intersection_area(ring_idx, x_min, x_max, y_min, y_max, _polys_valid);
			cell_value += material_value * fraction;
			fraction_sum += fraction; // for debugging
		}
	}
	if (_polys_valid && fabs(fraction_sum - 1.0) > 1e-12) {
		std::cerr << "ERROR SM2D::get_value: x=" << _dx*cell_k << " y=" << cell_j*_dy << 
			" fraction_sum = " << fraction_sum << "\n";
		for (auto polymat : _polymats) {
			std::cerr << "     polymat " << polymat << " " << bg::wkt(polymat->get_bpolys()) << "\n";
		}
		exit(-1);
	}
        
	return cell_value;
}

////////////////////////////////////////////////////////////////////////////////////
// ConstantMaterial3D
////////////////////////////////////////////////////////////////////////////////////
ConstantMaterial3D::ConstantMaterial3D(std::complex<double> value)
{
    _value = value;
}

std::complex<double> ConstantMaterial3D::get_value(double k, double j, double i)
{
    return _value;
}

void ConstantMaterial3D::get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2,
                                    int i1, int i2, double sx, double sy, double sz)
{
    int N = k2 - k1,
        M = j2 - j1;

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                grid((i-i1)*N*M + (j-j1)*N + k-k1) = _value;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////
// StructuredMaterial3D
////////////////////////////////////////////////////////////////////////////////////

void StructuredMaterial3D::initialize_class()
{
	cuda_total_polymats = 0;
	cuda_total_rings = 0;
	cuda_total_points = 0;
}

StructuredMaterial3D::StructuredMaterial3D(double X, double Y, double Z,
                                           double dx, double dy, double dz) :
                                           _X(X), _Y(Y), _Z(Z), 
                                           _dx(dx), _dy(dy), _dz(dz)
{
    _background = 1.0;
    _use_cache = true;
    _cache_active = false;
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
	std::cout << "SM3D::add_polymat polymat=" << polymat <<
		", material=" << polymat->get_matval().real() <<
		", z1=" << z1 << " z2=" << z2 << "\n";
    // Dummy variables
    StructuredMaterial2D* layer;
    double znew[2] = {z1, z2},
           z = 0;

    // Get access to relevant lists
    auto itl = _layers.begin();
    auto itz = _zs.begin();
    
    std::list<StructuredMaterial2D*>::iterator itl_ins;
    std::list<double>::iterator itz_ins;

    // Make sure the layer has a thickness
    if(z1 == z2) {
        std::cout << "Warning in Structured3DMaterial: Provided layer has no \
                      thickness. It will be ignored." << std :: endl;

        return;
    }
    else if(z2 < z1) {
        std::cout << "Warning in Structured3DMaterial: Provided layer has negative \
                      thickness. It will be ignored." << std :: endl;

        return;
    }

    // If this is the first addition, things are simple
    if(itz == _zs.end()) {
        _zs.push_back(z1);
        _zs.push_back(z2);
        
        layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
        layer->add_polymat(polymat);
        _layers.push_back(layer);

        return;
    }

    // now we insert the beginning and end point of the layer one at a time, breaking
    // up or inserting new layers as necessary
    for(int i = 0; i < 2; i++) {
        z = znew[i];

        itz = _zs.begin();
        itl = _layers.begin();
        itz_ins = _zs.end();
        itl_ins = _layers.end();

        // figure out where the point is going to go
        while(itz != _zs.end()) {
            if(z >= *itz) {
                itz_ins = itz;
                itl_ins = itl;
            }
            itz++;
            if(itl != _layers.end())
                itl++;
        }

        // Three cases to consider: (1) point below stack (2) point above stack (3)
        // point in stack
        if(itz_ins == _zs.end()) {
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
            _layers.push_front(layer);
            _zs.push_front(z);
        }
        else if(itz_ins == --_zs.end() && z != *itz_ins) {
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
            _layers.push_back(layer);
            _zs.push_back(z);
        }
        else {
            // make sure the point to insert is not already in the stack
            if(z != *itz_ins) {
                layer = new StructuredMaterial2D(_X, _Y, _dx, _dy);
                layer->add_polymats( (*itl_ins)->get_polymats() );
                _layers.insert(itl_ins, layer);
                _zs.insert(++itz_ins, z);
            }
        }
    }

    // Finally, insert the supplied PolyMat into the desired locations
    itz = _zs.begin();
    itl = _layers.begin();

    // figure out where the point is going to go
    while(itl != _layers.end()) {
        z = (*itz);
        if(z >= z1 && z < z2) {
            (*itl)->add_polymat(polymat);
        }
        itz++;
        itl++;
    }

	itz=_zs.begin();
    for(itl = _layers.begin(); itl != _layers.end(); itl++) {
		std::cout << "layer at z=" << *itz++ << "...\n";
        std::list<PolyMat*> polymats = (*itl)->get_polymats();

        for(auto ip = polymats.begin(); ip != polymats.end(); ip++) {
            std::cout << "   " << *ip << "  area=" << (*ip)->get_area() << " mat=" << (*ip)->get_matval().real() << "\n";
			//std::cout << "        " << bg::wkt((*ip)->get_bpoly()) << "\n";
		}
    }
	std::cout << "...final z=" << *itz << "\n";

    // aaannnddd we're done!
}

void StructuredMaterial3D::finalize()
{
	for (auto layer : _layers) {
		layer->finalize();
	}
}

std::complex<double> StructuredMaterial3D::get_value(double k, double j, double i)
{
    double zmin = (i-0.5) * _dz,
           zmax = (i+0.5) * _dz;

    std::complex<double> value = 0.0,
                         mat_val;

    std::list<double>::iterator itz = _zs.begin(),
                                itz_next;
    auto itl = _layers.begin();
    auto itcv = _cached_values.begin();
    auto itcf = _cached_flags.begin(); 

    bool cached = false;
    int jc = 0, kc = 0;

    // Check if i is below the stack
    if(zmax <= *itz) {
        return _background;
    }

    if(zmax > *itz && zmin < *itz) {
        value = (*itz - zmin) / _dz * 1.0;
        zmin = *itz;
    }

    while(itl != _layers.end())
    {

        itz_next = std::next(itz);
        if(zmin >= *itz && zmax <= *itz_next)
        {
            if(_use_cache and _cache_active) {
                jc = int(j) - _cache_j0;
                kc = int(k) - _cache_k0;

                cached = (*itcf)(jc, kc);
                if(cached) {
                    mat_val = (*itcv)(jc, kc);
                }
                else {
                    mat_val = (*itl)->get_value(k, j);
                    (*itcv)(jc, kc) = mat_val;
                    (*itcf)(jc, kc) = true;
                }
            }
            else {
                mat_val = (*itl)->get_value(k, j);
            }
      

            value += (zmax - zmin) / _dz * mat_val;

            return value;
        }
        else if(zmin >= *itz && zmin < *itz_next && zmax > *itz_next)
        {
            if(_use_cache and _cache_active) {
                jc = int(j) - _cache_j0;
                kc = int(k) - _cache_k0;

                cached = (*itcf)(jc, kc);
                if(cached) {
                    mat_val = (*itcv)(jc, kc);
                }
                else {
                    mat_val = (*itl)->get_value(k, j);
                    (*itcv)(jc, kc) = mat_val;
                    (*itcf)(jc, kc) = true;
                }
            }
            else {
                mat_val = (*itl)->get_value(k, j);
            }

            value += (*itz_next - zmin) / _dz * mat_val;
            zmin = *itz_next;
        }

        itl++;
        itz++;
        itcv++;
        itcf++;
    }

    value += (zmax - zmin) / _dz * 1.0;
    return value;
}

// Note that this takes a 1D array!
void StructuredMaterial3D::get_values(ArrayXcd& grid, int k1, int k2, 
                                                      int j1, int j2, 
                                                      int i1, int i2, 
                                                      double sx, double sy, double sz)
{
    int index = 0,
        Nx = k2-k1,
        Ny = j2-j1;

    int Nl, Nc;

    // if caching is enabled, setup all of the cache arrays
    // We need two arrays per slab: one to keep track of which values are already cached
    // and one to actually store the cached values
    if(_use_cache) {
        Nl = _layers.size();
        Nc = _cached_values.size();
        if(Nc != Nl) {
            _cached_values.resize(Nl);
            _cached_flags.resize(Nl);
        }

        for(auto ic = _cached_values.begin(); ic != _cached_values.end(); ic++) {
            (*ic).setZero(j2-j1, k2-k1);
        }
        for(auto flag = _cached_flags.begin(); flag != _cached_flags.end(); flag++) {
            (*flag).setZero(j2-j1, k2-k1);
        }

        _cache_j0 = int(j1+sy);
        _cache_k0 = int(k1+sx);
        _cache_J = j2-j1;
        _cache_K = k2-k1;
        _cache_active = true;

    }

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                index = (i-i1)*Nx*Ny + (j-j1)*Nx + (k-k1);
                grid(index) = get_value(k+sx, j+sy, i+sz);
            }
        }
    }

    _cache_active = false;
}
