#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

#include <cmath>
#include <complex>
#include <string>
#include <vector>
#include <list>

#ifndef __GRID_CUDA_HPP__
#define __GRID_CUDA_HPP__

// Specify the precision used for computational geometry subroutines
// In many situations, double is sufficient.  However, cases may be
// encountered where quad-precision is required (this is why CGAL 
// supports arbitrary precision).  To be safe, we generally use quad-
// precision.
//#define GFLOAT __float128
//#define GFLOAT double

typedef boost::geometry::model::d2::point_xy<double> BoostPoint;
typedef boost::geometry::model::polygon<BoostPoint> BoostPolygon;
using namespace Eigen;

typedef Array<bool, Dynamic, Dynamic> ArrayXXb;

namespace Grid {

/* A polygon which defines the boundary of an entire Yee cell or an arbitrary portion of a Yee cell.
 *
 * Although we are using a discrete grid of Yee cells, it is possible to continuously vary 
 * the material constants within this Yee cell.  We can take advantage of this fact by
 * smoothing our grid in such a way that our matrix A changes continously with perturbations to
 * the system geometry.
 *
 * In order to make this smoothing accurate, it is convenient to maintain a true geometric 
 * representation of a Yee cells.  This allows us to compute the exact overlap between a 
 * polygon which defines the material structure of the system and a given Yee cell.
 */
	class GridCell {
	private:
		std::vector<BoostPolygon> _bpolys;

		double _area,
			_max_area;

	public:
		GridCell();
		
		void set_vertices(double xmin, double xmax, double ymin, double ymax);
		double intersect(const BoostPolygon bpoly);
		double get_area_ratio();		
	};

/* A solid Polygon primitive.
 *
 * A Polygon is defined by a list of points specified in clockwise or counterclockwise order.
 * Both concave and convex polygons are supported. The Polygon is intended to be a flexible
 * primitive that can handle both simple and complicated geometry.
 */
	class PolyMat {
	private:
		std::complex<double> _matval;
		BoostPolygon _bpoly;
        
	public:
		/* Constructor
		 * @x list of x positions of polygon vertices
		 * @y list of y positions of polygon vertices
		 * @n number of elements in x and y
		 * @mat the complex material value
		 */
	    PolyMat(double* x, double* y, int n, std::complex<double> matval);

		/* Constructor
		 * @verts boost vertices
		 * @mat the complex material value
		 */
	    PolyMat(BoostPolygon bpoly, std::complex<double> matval);

 		//- Destructor
		~PolyMat();

	    inline const BoostPolygon get_bpoly() { return _bpoly; };

		/* Determine whether a point in real space is contained within the PolyMat 
		 * @x the x coordinate (real space)
		 * @y the y coordinate (real space)
		 * @return true if the point (x,y) is contained within the PolyMat. False otherwise.
		 */
		bool contains_point(double x, double y);

		/* Get the PolyMat's material value.
		 * 
		 * Note: This does not check if (x,y) is contained in the PolyMat.  Use 
		 * <contains_point> first if that functionality is needed.
		 *
		 * @return the complex material value of the PolyMat.
		 */
	    inline std::complex<double> get_matval() { return _matval; };

	    inline double get_area() { return boost::geometry::area(_bpoly); };
		double get_cell_overlap(GridCell& cell);
	};

/* Material class which provides the foundation for defining the system materials/structure.
 *
 * A Material must satisfy perform one function: given a spatial index, a complex 
 * material value is returned.  This is accomplished by extending the Material class and
 * implementing the <get_value> function.
 */
/* A flexible <Material> which consists of layerd <PolyMats>.
 *
 * A StructuredMaterial consists of one or more PolyMats defined by the user 
 * which are arranged within the simulation region.  
 */
	class StructuredMaterial2D {
	private:
	    boost::geometry::model::box<BoostPoint> _envelope;
		std::list<PolyMat*> _polymats;

        std::complex<double> _value;

		double _X,
			_Y,
			_dx,
			_dy;
	public:

		/* Constructor
		 * @X the width of the simulation region
		 * @Y the height of the simulation region
		 * @dx the horizontal grid spacing of the simulation region
		 * @dy the vertical grid spacing of the simulation region
		 *
		 * The width, height, and grid spacing must be the same as those supplied when creating
		 * the corresponding FDFD object.  This is essential to mapping from real space to 
		 * array indexing when constructing the system matrix.
		 */
		StructuredMaterial2D(double X, double Y, double dx, double dy);

		//- Destructor
		~StructuredMaterial2D();
		
		/* Add a primitive object to the Material.
		 * @prim the primitive to add.
		 *
		 * References to primitives are stored in an internal vector.  Working with references
		 * are advantageous as it allows the user to modify the geometry with minimal fuss
		 * between simulations.  This, however, necessitates that the corresponding 
		 * <MaterialPrimitive> objects not go out of scope while the StructuredMaterial is
		 * still in use.
		 */
		void add_polymat(PolyMat* polymat);
        void add_polymats(std::list<PolyMat*> polymats);

		/* Get the complex material value at an indexed position.
		 * @x the x index (column) of the material value
		 * @y the y index (row) of the material value
		 * @return the complex material value at (x,y).  If no MaterialPrimitive exists at (x,y), 1.0 is returned.
		 */
		std::complex<double> get_value(double x, double y);

        void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, double sx, double sy);

        /* Get the list of primitives belonging to this StructuredMaterial
         * @return The std::list<MaterialPrimitive*> containing the constituent
         * MaterialPrimitives
         */
        inline std::list<PolyMat*> get_polymats() { return _polymats; };

	    void verify_area();
	};

/* Material class which provides the foundation for defining the system materials/structure.
 *
 * A Material must satisfy perform one function: given a spatial index, a complex 
 * material value is returned.  This is accomplished by extending the Material class and
 * implementing the <get_value> function.
 */
	class Material3D {
	
	public:
		/* Query the material value at a point in real space.
		 * @x The x index of the query
		 * @y The y index of the query
		 *
		 * The structure of the electromagnetic system being solved is ultimately defined
		 * in terms of spatially-dependent materials. The material is defined on a 
		 * spatial grid which is directly compatible with finite differences.
		 * See <StructuredMaterial> for specific implementations.
		 *
		 * @return the complex material at position (x,y).
		 */
		virtual std::complex<double> get_value(double k, double j, double i) = 0;

		/* Get a block of values.
		 */
		virtual void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2, 
								int i1, int i2, double sx, double sy, double sz) = 0;
		virtual ~Material3D() {};
	};

/* A 3D material distribution defined by a single constant value.
 *
 * Use this for uniform materials.
 */
	class ConstantMaterial3D : public Material3D {
	private:
		std::complex<double> _value;
	
	public:
		ConstantMaterial3D(std::complex<double> value);

		/* Query the material value at a point in real space.
		 *
		 * This will always return the same value
		 *
		 * @x The x index of the query
		 * @y The y index of the query
		 * @return the complex material
		 */
		std::complex<double> get_value(double k, double j, double i);

		/* Get a block of values.
		 *
		 * This just fills the provided array with a single value
		 */
		void get_values(ArrayXcd& grid, int k1, int k2, int j1, int j2,
						int i1, int i2, double sx, double sy, double sz);

	}; // ConstantMaterial3D

/* Define a 3D planar stack structure.
 *
 * This class is essentially an extension of the structured3DMaterial to three
 * dimensions. It is built up of StructuredMaterials which have a defined position
 * and thickness in the z-direction. This allows the used to define 3D grid-smoothed
 * structures that have a slab-like construction (which is most common in the micro-
 * and nanoscale worlds).
 */
	class StructuredMaterial3D : public Material3D {
	private:
        std::list<StructuredMaterial2D*> _layers;
        std::list<double> _zs;

		double _X,
			_Y,
			_Z,
			_dx,
			_dy,
			_dz,
			_background;


        // cache-related parameters
        std::list<ArrayXXcd> _cached_values;
        std::list<ArrayXXb> _cached_flags;

        bool _use_cache, _cache_active;
        
        int _cache_j0, _cache_k0,
            _cache_J,  _cache_K;

	public:

		/* Constructor
		 * @X the width of the simulation region in x
		 * @Y the width of the simulation region in y
		 * @Z the width of the simulation region in z
		 * @dx the x grid spacing of the simulation region
		 * @dy the y grid spacing of the simulation region
		 * @dz the z grid spacing of the simulation region
		 *
		 * The width, height, and grid spacing must be the same as those supplied when creating
		 * the corresponding FDFD object.  This is essential to mapping from real space to 
		 * array indexing when constructing the system matrix.
		 */
		StructuredMaterial3D(double X, double Y, double Z, double dx, double dy, double dz);

		//- Destructor
		~StructuredMaterial3D();
		
		/* Add a primitive object to the Material.
		 * @prim the primitive to add.
		 * @z1 the lower z bound of the primitive
		 * @z2 the upper z bound of the primitive
		 *
		 * References to primitives are stored in an internal vector.  Working with references
		 * are advantageous as it allows the user to modify the geometry with minimal fuss
		 * between simulations.  This, however, necessitates that the corresponding 
		 * <MaterialPrimitive> objects not go out of scope while the StructuredMaterial is
		 * still in use.
		 */
		void add_polymat(PolyMat* polymat, double z1, double z2);

		/* Get the complex material value at an indexed position.
		 * @x the x index (column) of the material value
		 * @y the y index (row) of the material value
		 * @return the complex material value at (x,y).  If no MaterialPrimitive exists at (x,y), 1.0 is returned.
		 */
		std::complex<double> get_value(double k, double j, double i);

        void get_values(ArrayXcd& grid, int k1, int k2, 
						int j1, int j2, 
						int i1, int i2, 
						double sx=0, double sy=0, double sz=0);

	};

}; // grid namespace

#endif