#ifndef VOLUME_H
#define VOLUME_H

#include "grid.h"
#include "vector3.h"
#include "material.h"

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
	virtual Vector3* fetchGradient() = 0;

	float _dx, _dy, _dz;	// volume size
	float _world_to_local_x, _world_to_local_y, _world_to_local_z;	// scaling factor
	float _local_to_world_x, _local_to_world_y, _local_to_world_z;
	Vector3 _min_coord;
	Vector3 _max_coord;

	double _oa, _ot, _os;	// oa: absorption, ot: extinction(attenuation), os: scatter
	double _wds; // ray caster sample step
	Grid *_grid;
	Material *_mat;
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

	virtual Vector3* fetchGradient() override {
		Vector3 *gradient = new Vector3[_grid->_xdim * _grid->_ydim * _grid->_zdim];
		float denom_x = 0.5 * (_dx / (float)_grid->_xdim);
		float denom_y = 0.5 * (_dy / (float)_grid->_ydim);
		float denom_z = 0.5 * (_dz / (float)_grid->_zdim);

		float edge_denom_x = _dx / (float)_grid->_xdim;
		float edge_denom_y = _dy / (float)_grid->_ydim;
		float edge_denom_z = _dz / (float)_grid->_zdim;

		float dfx, dfy, dfz;
		for (int x = 1; x < _grid->_xdim-1; x++) {
			for (int y = 1; y < _grid->_ydim-1; y++) {
				for (int z = 1; z < _grid->_zdim-1; z++) {
					int index = _grid->_xdim * _grid->_ydim * z + _grid->_xdim * y + x;
					//if ((x - 1) < 0 || (y - 1) < 0 || (z - 1) < 0) {
					//	// at edge, use forward difference
					//	dfx = _grid->_data[_grid->gridIndexAt(x + 1, y, z)] - _grid->_data[_grid->gridIndexAt(x, y, z)];
					//	dfy = _grid->_data[_grid->gridIndexAt(x, y + 1, z)] - _grid->_data[_grid->gridIndexAt(x, y, z)];
					//	dfz = _grid->_data[_grid->gridIndexAt(x, y, z + 1)] - _grid->_data[_grid->gridIndexAt(x, y, z)];
					//	gradient[index].x = edge_denom_x * dfx;
					//	gradient[index].y = edge_denom_y * dfy;
					//	gradient[index].z = edge_denom_z * dfz;
					//}
					//else if (x == _grid->_xdim || y == _grid->_ydim || z == _grid->_zdim) {
					//	// at edge, use backward difference
					//	dfx = _grid->_data[_grid->gridIndexAt(x, y, z)] - _grid->_data[_grid->gridIndexAt(x - 1, y, z)];
					//	dfy = _grid->_data[_grid->gridIndexAt(x, y, z)] - _grid->_data[_grid->gridIndexAt(x, y - 1, z)];
					//	dfz = _grid->_data[_grid->gridIndexAt(x, y, z)] - _grid->_data[_grid->gridIndexAt(x, y, z - 1)];
					//	gradient[index].x = edge_denom_x * dfx;
					//	gradient[index].y = edge_denom_y * dfy;
					//	gradient[index].z = edge_denom_z * dfz;
					//}
					//else {
						// inside, use central difference
						// 1/2h * [f(x+h) - f(x-h)]
						dfx = _grid->_data[_grid->gridIndexAt(x + 1, y, z)] - _grid->_data[_grid->gridIndexAt(x - 1, y, z)];
						dfy = _grid->_data[_grid->gridIndexAt(x, y + 1, z)] - _grid->_data[_grid->gridIndexAt(x, y - 1, z)];
						dfz = _grid->_data[_grid->gridIndexAt(x, y, z + 1)] - _grid->_data[_grid->gridIndexAt(x, y, z - 1)];
						gradient[index].x = denom_x * dfx;
						gradient[index].y = denom_y * dfy;
						gradient[index].z = denom_z * dfz;
					//}
				}
			}
		}
		return gradient;
	}
};

#endif