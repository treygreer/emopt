#include "Grid_cuda.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace GridCuda;

class Line {
private:
	double _a;
	double _b;
	double _c;
public:
	Line(BoostPoint p1, BoostPoint p2) {
		_a = p2.y() - p1.y();
		_b = p1.x() - p2.x();
		_c = p2.x()*p1.y() - p1.x()*p2.y();
	}

	bool point_inside(BoostPoint p) {
        return _a*p.x() + _b*p.y() + _c >= 0.0;
	}

	bool intersect(const Line l, BoostPoint *p) {
		double ap = _b*l._c - l._b*_c;
		double bp = l._a*_c - _a*l._c;
		double cp = _a*l._b - l._a*_b;
		if (cp != 0) {
			p->x(ap/cp);
			p->y(bp/cp);
			return true;
		} else {
			p->x(NAN);
			p->y(NAN);
			return false;
		}
	}
};

double trapezoid_box_intersection_area(const BoostRing trapezoid, const BoostBox box)
{
	//std::cerr << "trap box intersection area: \n";
	//std::cerr << "   trapezoid:  " << bg::wkt(trapezoid) << "\n";
	//std::cerr << "   box:  " << bg::wkt(box) << "\n";
	BoostRing box_ring;
	bg::convert(box, box_ring);
	
    // from https://en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm
	BoostRing output_ring;
	bg::assign(output_ring, trapezoid);
	for (auto clip_seg=bg::segments_begin(box_ring); clip_seg!=bg::segments_end(box_ring); ++clip_seg) {
		Line clip_line(*(clip_seg->first), *(clip_seg->second));
		BoostRing input_ring;
		bg::assign(input_ring, output_ring);
		//std::cerr << "       clip_seg:  " << bg::wkt(*clip_seg) << "\n";
		//std::cerr << "       input_ring:  " << bg::wkt(input_ring) << "\n";
		output_ring.clear();
		for (auto input_seg=bg::segments_begin(input_ring); input_seg!=bg::segments_end(input_ring); ++input_seg) {
			BoostPoint prev_point = *(input_seg->first);
			BoostPoint current_point = *(input_seg->second);
			Line input_line(prev_point, current_point);
			BoostPoint intersecting_point;
			clip_line.intersect(input_line, &intersecting_point);
			//std::cerr << "           input_seg:  " << bg::wkt(*input_seg) << "\n";
			//std::cerr << "           prev_point:  " << bg::wkt(prev_point) << " inside=" << clip_line.point_inside(prev_point) << "\n";
			//std::cerr << "           current_point:  " << bg::wkt(current_point) << " inside=" << clip_line.point_inside(current_point) << "\n";
			//std::cerr << "           intersecting_point:  " << bg::wkt(intersecting_point) << "\n";
			if (clip_line.point_inside(current_point)) { 
				if (!clip_line.point_inside(prev_point)) {
					bg::append(output_ring, intersecting_point);
				} 
				bg::append(output_ring, current_point);
			} else {
				if (clip_line.point_inside(prev_point)) {
					bg::append(output_ring, intersecting_point);
				}
			} 
		}
		bg::correct(output_ring);  // close ring
	}
	//std::cerr << "   output_ring:  " << bg::wkt(output_ring) << "\n";
	//std::cerr << "   area = " << bg::area(output_ring) << "\n";
	return bg::area(output_ring);
}

double ring_box_intersection_area(BoostRing ring, BoostBox box, bool debug)
{
	double trapezoid_area_sum = 0.0;
	double xmin_box = bg::get<bg::min_corner,0>(box);
	for (auto it=bg::segments_begin(ring); it!=bg::segments_end(ring); ++it) {
		BoostPoint p0=*(it->first), p1=*(it->second);
		double xmin_pts = std::min(bg::get<0>(p0), bg::get<0>(p1));
		double xmin = std::min(xmin_pts, xmin_box) - 1.0;
		double y0=bg::get<1>(p0);
		double y1=bg::get<1>(p1);
		BoostRing trapezoid { {xmin, y0}, p0, p1, {xmin, y1} };
		bg::correct(trapezoid);
		double intersection_area = trapezoid_box_intersection_area(trapezoid, box);
		if (y0 < y1) intersection_area = -intersection_area;
		trapezoid_area_sum += intersection_area;
	}

	BoostMultiPolygon intersection_bpolys;
	bg::intersection(ring, box, intersection_bpolys);
	double boost_area = bg::area(intersection_bpolys);
	if (fabs(boost_area-trapezoid_area_sum) / boost_area > 1e-12 &&
		fabs(boost_area-trapezoid_area_sum) > 1e-12) 
	{
		std::cerr << "boost_area=" << boost_area << "  trapezoid_area_sum=" << trapezoid_area_sum << "\n";
		std::cerr << "  ring: " << bg::wkt(ring) << "\n";
		std::cerr << "  box:  " << bg::wkt(box) << "\n";
		std::cerr << "  intersection:  " << bg::wkt(intersection_bpolys) << "\n";
		//exit(-1);
	}
	return trapezoid_area_sum;
	//return boost_area;
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
StructuredMaterial2D::StructuredMaterial2D(double X, double Y, double dx, double dy,
										   double background) :
	_X(X), _Y(Y), _dx(dx), _dy(dy), _background(background)
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


std::complex<double> StructuredMaterial2D::get_value(double x, double y)
{
	std::complex<double> value = 0.0;
	double xmin=(x-0.5)*_dx, xmax=(x+0.5)*_dx;
	double ymin=(y-0.5)*_dy, ymax=(y+0.5)*_dy;
	BoostBox cell_bbox = BoostBox(BoostPoint(xmin, ymin), BoostPoint(xmax, ymax));
	double inv_cell_area = 1.0 / bg::area(cell_bbox);

	double fraction_sum = 0.0;
	for (auto polymat : _polymats) {
		for (auto bpoly : polymat->get_bpolys()) {
			double outer_fraction = inv_cell_area * ring_box_intersection_area(bpoly.outer(), cell_bbox,
				_polys_valid);
			fraction_sum += outer_fraction; // for debugging
			value += polymat->get_matval() * outer_fraction;

			for (auto inner_ring : bpoly.inners()) {
				double inner_fraction = inv_cell_area * ring_box_intersection_area(inner_ring, cell_bbox,
					_polys_valid);
				fraction_sum -= inner_fraction; // for debugging
				value -= polymat->get_matval() * inner_fraction;
			}
		}
	}
	if (_polys_valid && fabs(fraction_sum - 1.0) > 1e-12) {
		std::cerr << "ERROR SM2D::get_value: x=" << x << " y=" << y << " fraction_sum = " << fraction_sum << "\n";
		std::cerr << "     cell_bbox " << bg::wkt(cell_bbox) << "\n";
		for (auto polymat : _polymats) {
			std::cerr << "     polymat " << polymat << " " << bg::wkt(polymat->get_bpolys()) << "\n";
		}
		exit(-1);
	}
        
	return value;
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
                                    int i1, int i2,
									double xoff, double yoff, double zoff)
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

StructuredMaterial3D::StructuredMaterial3D(double X, double Y, double Z,
                                           double dx, double dy, double dz,
	                                       double background) :
                                           _X(X), _Y(Y), _Z(Z), 
                                           _dx(dx), _dy(dy), _dz(dz),
										   _background(background)
{
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
        
        layer = new StructuredMaterial2D(_X, _Y, _dx, _dy, _background);
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
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy, _background);
            _layers.push_front(layer);
            _zs.push_front(z);
        }
        else if(itz_ins == --_zs.end() && z != *itz_ins) {
            layer = new StructuredMaterial2D(_X, _Y, _dx, _dy, _background);
            _layers.push_back(layer);
            _zs.push_back(z);
        }
        else {
            // make sure the point to insert is not already in the stack
            if(z != *itz_ins) {
                layer = new StructuredMaterial2D(_X, _Y, _dx, _dy, _background);
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
        value = (*itz - zmin) / _dz * _background;
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

    value += (zmax - zmin) / _dz * _background;
    return value;
}

// Note that this takes a 1D array!
void StructuredMaterial3D::get_values(ArrayXcd& grid,
									  int k1, int k2, 
									  int j1, int j2, 
									  int i1, int i2, 
									  double xoff, double yoff, double zoff)
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

        for(auto& cv : _cached_values) {
            cv.setZero(j2-j1, k2-k1);
        }
        for(auto& cf : _cached_flags) {
            cf.setZero(j2-j1, k2-k1);
        }

        _cache_j0 = int(j1+yoff);
        _cache_k0 = int(k1+xoff);
        _cache_J = j2-j1;
        _cache_K = k2-k1;
        _cache_active = true;

    }

    for(int i = i1; i < i2; i++) {
        for(int j = j1; j < j2; j++) {
            for(int k = k1; k < k2; k++) {
                index = (i-i1)*Nx*Ny + (j-j1)*Nx + (k-k1);
                grid(index) = get_value(k+xoff, j+yoff, i+zoff);
            }
        }
    }

    _cache_active = false;
}
