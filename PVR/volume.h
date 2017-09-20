#ifndef VOLUME_H
#define VOLUME_H

#include "grid.h"
#include "vector3.h"

/* Volume grid is an abstract volume, proportional to the actual grid size. 
 * Volume instance has a pointer to its corresponding grid.
 * Data are stored in grid, while position is represented by volume.
 */
class Volume {
public:
	Volume() {
		_grid = nullptr;
	}

	double transparency() {
		return exp(-_ot*_wds);
	}
	void setCoefficent(double oa, double ot, double os) {
		_oa = oa; _ot = ot; _os = os;
	}

	virtual double radiance(double lambda, double T) = 0;
	virtual Vector3 pos_map(Vector3 p) = 0;

	float _dx, _dy, _dz;	// volume size
	float _world_to_local_x, _world_to_local_y, _world_to_local_z;	// scaling factor
	float _local_to_world_x, _local_to_world_y, _local_to_world_z;
	Vector3 _min_coord;
	Vector3 _max_coord;

	double _oa, _ot, _os;	// oa: absorption, ot: extinction(attennuation), os: scatter
	double _wds; // ray caster sample step
	Grid *_grid;
};


class BlackBody : public Volume {
public:
	BlackBody() {
		_grid = new Grid();
	}
	BlackBody(Grid* g) {
		_grid = g;
		int dmax = g->_xdim;
		if (dmax < g->_ydim) dmax = g->_ydim;
		if (dmax < g->_zdim) dmax = g->_zdim;
		_dx = (float)g->_xdim / (float)dmax;
		_dy = (float)g->_ydim / (float)dmax;
		_dz = (float)g->_zdim / (float)dmax;
		_world_to_local_x = _dx / (float)g->_xdim;
		_world_to_local_y = _dy / (float)g->_ydim;
		_world_to_local_z = _dz / (float)g->_zdim;
		_local_to_world_x = 1.0 / _world_to_local_x;
		_local_to_world_y = 1.0 / _world_to_local_y;
		_local_to_world_z = 1.0 / _world_to_local_z;
	}

	const double C_1 = 3.7418e-16;
	const double C_2 = 1.4388e-2;
	const double LeScale = 0.25;

	/* set bbox of volume */
	void setBox(Vector3 mincoord, Vector3 maxcoord) {
		_min_coord.x = mincoord.x * _world_to_local_x;
		_min_coord.y = mincoord.y * _world_to_local_y;
		_min_coord.z = mincoord.z * _world_to_local_z;
		_max_coord.x = maxcoord.x * _world_to_local_x;
		_max_coord.y = maxcoord.y * _world_to_local_y;
		_max_coord.z = maxcoord.z * _world_to_local_z;
	}

	virtual double radiance(double lambda, double T) override {
		return (2.0*C_1) / (pow(lambda, 5.0) * (exp(C_2 / (lambda*T)) - 1.0));
	}

	virtual Vector3 pos_map(Vector3 p) override {
		/* Map volume position to grid position. */
		return Vector3(_local_to_world_x*p.x, _local_to_world_y*p.y, _local_to_world_z*p.z);
	}
};

#endif