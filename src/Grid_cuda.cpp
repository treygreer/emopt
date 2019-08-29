#include "Grid_cuda.hpp"
#include "CudaClipper.hpp"
#include <iostream>
#include <climits>
#include <ctime>
#include <exception>
#include <omp.h>

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


// Note that this takes a 1D array!
void StructuredMaterial3D::get_values(std::complex<double>* grid,
									  int k1, int k2, 
									  int j1, int j2, 
									  int i1, int i2, 
									  double koff, double joff, double ioff)
{
	auto layer = _layers.begin();
	auto layer_next = _layers.begin()++;
	CudaClipper cuda_clipper(k1, k2, j1, j2, i1, i2, koff, joff, _dx, _dy);
	StructuredMaterialLayer* values_layer = NULL;

    for(int slice_idx = i1; slice_idx < i2; slice_idx++) {  // z index

		double       slice_z_min = (slice_idx+ioff-0.5) * _dz;
		const double slice_z_max = (slice_idx+ioff+0.5) * _dz;

		while (layer_next != _layers.end()) {
			double layer_z_base = (*layer)->z_base();
			double layer_z_top = (*layer_next)->z_base();

			if (slice_z_min >= layer_z_base) {
				if (values_layer != *layer) {
					cuda_clipper.compute_layer_values(*layer);
					values_layer = *layer;
				}
				if (slice_z_max <= layer_z_top) {
					cuda_clipper.composite_layer_values_into_slice((slice_z_max - slice_z_min) / _dz,
																slice_idx);
					break;  // break out of layer loop: stay on this layer and go to next slice
				}
				else if (slice_z_min < layer_z_top) {
					assert(slice_z_max > layer_z_top);
					cuda_clipper.composite_layer_values_into_slice((layer_z_top - slice_z_min) / _dz,
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

	cuda_clipper.return_grid_values(grid);
}
