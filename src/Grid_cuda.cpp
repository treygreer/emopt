#include "Grid_cuda.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

using namespace Grid;

/**************************************** Materials ****************************************/

//------------------------------ MaterialPrimitives ------------------------------------/
GridCell::GridCell()
{
}
		
void GridCell::set_vertices(double xmin, double xmax, double ymin, double ymax)
{
	double area;
	_bpolys.clear();

	BoostPolygon new_bpoly;
	boost::geometry::append(new_bpoly, BoostPoint(xmin, ymin));
	boost::geometry::append(new_bpoly, BoostPoint(xmin, ymax));
	boost::geometry::append(new_bpoly, BoostPoint(xmax, ymax));
	boost::geometry::append(new_bpoly, BoostPoint(xmax, ymin));
    boost::geometry::correct(new_bpoly);

    _bpolys.push_back(new_bpoly);
	area = boost::geometry::area(new_bpoly);
	_area = fabs(area);
	_max_area = _area;
}

double GridCell::intersect(const BoostPolygon bpoly)
{
	double area = 0.0,
		   intersected_area,
           geo_area;

	std::vector<BoostPolygon> bdiffs;
	
    // Do the difference
	for(auto bp : _bpolys) {
		boost::geometry::difference(bp, bpoly, bdiffs);
	}
	_bpolys.clear();
	
	for(auto bd : bdiffs) {
		_bpolys.push_back(bd);
        geo_area = boost::geometry::area(bd);
		area += fabs(geo_area);
	}
	intersected_area = _area - area;
	_area = area;
	return intersected_area/_max_area;

}

double GridCell::get_area_ratio()
{
	return _area/_max_area;
}		


//------------------------------ Polygon ------------------------------------/

PolyMat::PolyMat(double* x, double* y, int n, std::complex<double> matval) :
	_matval(matval)
{
    for(int i = 0; i < n; i++) {
        boost::geometry::append(_bpoly, boost::geometry::make<BoostPoint>(x[i], y[i]));
    }

    // correct the geometry
    boost::geometry::correct(_bpoly);

	boost::geometry::model::box<BoostPoint> bbox;
    boost::geometry::envelope(_bpoly, bbox);
	std::cout << "new polygon "<< this << "... matval=" << matval.real() <<
		", bbox=" << boost::geometry::dsv(bbox) <<
		", area=" << boost::geometry::area(_bpoly) << "\n";
}

PolyMat::PolyMat(BoostPolygon bpoly, std::complex<double> matval) :
	_matval(matval), _bpoly(bpoly)
{
}

PolyMat::~PolyMat()
{
	_bpoly.clear();
}

bool PolyMat::contains_point(double x, double y)
{
    BoostPoint bp(x, y);
	return boost::geometry::within(bp, _bpoly);
}

double PolyMat::get_cell_overlap(GridCell& cell)
{
	return cell.intersect(_bpoly);
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
    boost::geometry::envelope(background_polymat->get_bpoly(), _envelope);

    _polymats.push_front(background_polymat);
	std::cout << "SM2D::new  area=" << boost::geometry::area(_envelope) << " background_polymat=" << background_polymat << "\n";
}

StructuredMaterial2D::~StructuredMaterial2D() {}

/* It is important to the material averaging algorithm that polygons be stored in an 
 * ordered list according to their layer.  Lower layers are stored first (have priority).
 * This means that once you have added a polygon to a list, you cannot change its
 * layer!
 */
void StructuredMaterial2D::add_polymat(PolyMat* polymat)
{
	std::list<BoostPolygon> clipped_input_bpolys;
	// clip polymat to material envelope (useful for area assertions)
	boost::geometry::intersection(polymat->get_bpoly(), _envelope, clipped_input_bpolys);
	
	// clip preceeding polymats with new clipped polymats
	std::list<BoostPolygon> diffs;
	std::list<PolyMat*> new_polymats;
	for (auto bit : clipped_input_bpolys) {
		for (auto it : _polymats) {
			boost::geometry::difference(it->get_bpoly(), bit, diffs);
			for (auto dit : diffs) {
				PolyMat *new_polymat = new PolyMat(dit, it->get_matval());
				new_polymats.push_back(new_polymat);
			}
		}
	}
	_polymats.clear();
	_polymats = new_polymats;

	// add new polygons to list
	for (auto bit : clipped_input_bpolys) {
		PolyMat *new_polymat = new PolyMat(bit, polymat->get_matval());
		_polymats.push_back(new_polymat);
	}

	this->verify_area();
}

void StructuredMaterial2D::verify_area()
{
	double total_area = 0.0;
	for (auto poly : _polymats) {
		double poly_area = boost::geometry::area(poly->get_bpoly());
		assert(poly_area > 0.0);
		total_area += poly_area;
	}
	assert (total_area == boost::geometry::area(_envelop));
}

void StructuredMaterial2D::add_polymats(std::list<PolyMat*> polygons)
{
    std::list<PolyMat*>::iterator it;
    for (it = polygons.begin(); it != polygons.end(); it++) {
        add_polymat(*it);
    }
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

// This attempts to compute a reasonable average of the materials in a given Yee cell
// Note that there are a few situations where this average will not quite be what they
// should be.  In particular, if three or more materials intersect a cell, this 
// average will begin to deviate from the "correct" average
std::complex<double> StructuredMaterial2D::get_value(double x, double y)
{
	std::complex<double> val = 0.0;
	std::list<PolyMat*>::iterator it = _polymats.begin();
	PolyMat* poly;
	GridCell cell;
	
	double xd = x*_dx, //+ _dx/2.0,
		   yd = y*_dy; //+ _dy/2.0;

	double xmin = xd - _dx/2.0,
		   xmax = xd + _dx/2.0,
		   ymin = yd - _dy/2.0,
		   ymax = yd + _dy/2.0,
		   overlap = 1.0;

    bool contains_p1,
         contains_p2,
         contains_p3,
         contains_p4;

	cell.set_vertices(xmin,xmax,ymin,ymax);
	
	if(_polymats.size() == 0) {
		std::cerr << "Error: StructuredMaterial list is empty." << std::endl;
		return 0.0;
	}

	//std::cout << "------------------------" << std::endl;
	while(it != _polymats.end()) {
		poly = (*it);
        
        // These values are used twice, so we recompute them
        contains_p1 = poly->contains_point(xmin,ymin);
        contains_p2 = poly->contains_point(xmax,ymin);
        contains_p3 = poly->contains_point(xmax,ymax);
        contains_p4 = poly->contains_point(xmin,ymax);
		
		if(contains_p1 && contains_p2 &&
		   contains_p3 && contains_p4 &&
		   cell.get_area_ratio() == 1.0) 
		{
				return poly->get_matval();
		}
		else if(contains_p1 || contains_p2 ||
		        contains_p3 || contains_p4) 
		{
			overlap = poly->get_cell_overlap(cell);

			val += overlap * poly->get_matval();
		}
		it++;

		if(cell.get_area_ratio() == 0) {
			break;
		}

	}

	// assume background has index of 1.0
	if(cell.get_area_ratio() > 0) {
		val += cell.get_area_ratio()*1.0;
	}

	return val;
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
	for(auto it = _layers.begin(); it != _layers.end(); it++) {
        delete (*it);
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
        printf("layer at z=%f...\n", *itz++);
        std::list<PolyMat*> polys = (*itl)->get_polymats();

        for(auto ip = polys.begin(); ip != polys.end(); ip++) {
            std::cout << "   " << *ip << "  area=" << (*ip)->get_area() << " mat=" << (*ip)->get_matval().real() << "\n";
			//std::cout << "        " << boost::geometry::dsv((*ip)->get_bpoly()) << "\n";
		}
    }
    printf("...final z=%f\n", *itz);

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
