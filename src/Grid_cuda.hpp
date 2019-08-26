
// ignore particular warning in boost/geometry
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/geometry.hpp>
#pragma GCC diagnostic pop

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

namespace GridCuda {

	namespace bg = ::boost::geometry;
	typedef bg::model::d2::point_xy<double> BoostPoint;
	typedef bg::model::polygon<BoostPoint> BoostPolygon;
	typedef bg::model::multi_polygon<BoostPolygon> BoostMultiPolygon;
	typedef bg::model::ring<BoostPoint> BoostRing;
	typedef bg::model::box<BoostPoint> BoostBox;
	typedef bg::model::segment<BoostPoint> BoostSegment;

/* A solid Polygon primitive.
 *
 * A Polygon is defined by a list of points specified in clockwise or counterclockwise order.
 * Both concave and convex polygons are supported. The Polygon is intended to be a flexible
 * primitive that can handle both simple and complicated geometry.
 */
	class PolyMat {
	private:
		std::complex<double> _matval;
		BoostMultiPolygon _bpolys;
        
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
	    PolyMat(BoostMultiPolygon bpolys, std::complex<double> matval);

		/* Copy Constructor
		 */
	    PolyMat(PolyMat *pm);

 		//- Destructor
		~PolyMat();

	    inline const BoostMultiPolygon get_bpolys() { return _bpolys; };

		/* Get the PolyMat's material value.
		 * 
		 * Note: This does not check if (x,y) is contained in the PolyMat. 
		 *
		 * @return the complex material value of the PolyMat.
		 */
	    inline std::complex<double> get_matval() { return _matval; };

		void clip(BoostBox box);
		void subtract(BoostMultiPolygon bpolys);

	    inline double get_area() { return bg::area(_bpolys); };
		inline bool is_empty() { return bg::is_empty(_bpolys); };
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
	class StructuredMaterialLayer {
	private:
	    BoostBox _envelope;
		std::list<PolyMat*> _polymats;
		bool _polys_valid;  // used to condition area-based error reporting

		double _X, _Y, _dx, _dy, _background;  // from parent SM3D
		double _z_base;  
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
		StructuredMaterialLayer(double X, double Y, double dx, double dy, double background,
			double z_base);

		//- Destructor
		~StructuredMaterialLayer();
		
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

        /* Get the list of primitives belonging to this StructuredMaterial
         * @return The std::list<MaterialPrimitive*> containing the constituent
         * MaterialPrimitives
         */
        inline std::list<PolyMat*> get_polymats() { return _polymats; };

	    void verify_area();

		inline double z_base() { return _z_base; };
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
		 */

		/* Get a block of values.
		 */
		virtual void get_values(std::complex<double>* grid,
								int k1, int k2, int j1, int j2, int i1, int i2,
								double koff, double joff, double ioff) = 0;
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

		/* Get a block of values.
		 *
		 * This just fills the provided array with a single value
		 */
		void get_values(std::complex<double>* grid,
						int k1, int k2, int j1, int j2, int i1, int i2,
						double koff, double joff, double ioff);

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
        std::list<StructuredMaterialLayer*> _layers;

		double _X,
			_Y,
			_Z,
			_dx,
			_dy,
			_dz,
			_background;

		static constexpr double _MIN_Z = -1E300;
		static constexpr double _MAX_Z = 1E300;

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
		StructuredMaterial3D(double X, double Y, double Z, double dx, double dy, double dz,
			double background=1.0);

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

		/* Get the complex material value at the indexed positions.
		 * @return the complex material values at (k,j,i).  If no MaterialPrimitive exists at (x,y), _background is returned.
		 */
        void get_values(std::complex<double>* grid,
						int k1, int k2, 
						int j1, int j2, 
						int i1, int i2, 
						double koff=0, double joff=0, double ioff=0);
	};

}; // GridCuda namespace

#endif
