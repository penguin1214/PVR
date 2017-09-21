#ifndef GRID_H
#define GRID_H

#include "Vector3.h"

#include "externals/glm/gtx/transform.hpp"

class Grid {
public:
	Grid() {
		_trans = new double[16]; _itrans = new double[16];
	}
	Grid(int xd, int yd, int zd) : _xdim(xd), _ydim(yd), _zdim(zd) {
		_trans = new double[16]; _itrans = new double[16];
	}

	float *_data;
	int _xdim = 30;
	int _ydim = 60;
	int _zdim = 30;
	int GRID_SIZE = 4;	// ???

	Vector3 _min_coord = Vector3(0.0, 0.0, 0.0);
	Vector3 _max_coord = Vector3(0.0, 0.0, 0.0);

	glm::dmat4x4 _mat_trans;
	glm::dmat4x4 _mat_itrans;
	double *_trans, *_itrans;	// world <-> local transform

	void setSize(int xd, int yd, int zd) { _xdim = xd; _ydim = yd; _zdim = zd; }
	__host__ __device__ int gridIndexAt(int i, int j, int k) const {
		return i + (_xdim)*j + (_xdim*_ydim*k);
	}

	__host__ __device__ void indexToLocal(int i, int j, int k, double &l_x, double &l_y, double &l_z) const {
		l_x = double(i) / double(_xdim - 1);
		l_y = double(j) / double(_ydim - 1);
		l_z = double(k) / double(_zdim - 1);
	}

	__host__ __device__ void localToWorld(double *trans, double l_x, double l_y, double l_z, double &w_x, double &w_y, double &w_z) const {
		w_x = l_x*trans[0] + l_y*trans[1] + l_z*trans[2] + trans[3];
		w_y = l_x*trans[4] + l_y*trans[5] + l_z*trans[6] + trans[7];
		w_z = l_x*trans[8] + l_y*trans[9] + l_z*trans[10] + trans[11];
	}

	__host__ __device__ void indexToWorld(double *trans, int i, int j, int k, double &w_x, double &w_y, double &w_z) const {
		double l_x, l_y, l_z;
		indexToLocal(i, j, k, l_x, l_y, l_z);
		localToWorld(trans, l_x, l_y, l_z, w_x, w_y, w_z);
	}

	__host__ __device__ void worldToLocal(double *itrans, const double w_x, const double w_y, const double w_z, double &l_x, double &l_y, double &l_z) const {
		l_x = w_x*itrans[0] + w_y*itrans[1] + w_z*itrans[2] + itrans[3];
		l_y = w_x*itrans[4] + w_y*itrans[5] + w_z*itrans[6] + itrans[7];
		l_z = w_x*itrans[8] + w_y*itrans[9] + w_z*itrans[10] + itrans[11];
	}

	__host__ __device__ void localToUpperLeftIndex(const double l_x, const double l_y, const double l_z, int &i, int &j, int &k) const {
		i = (int)floor(l_x*double(_xdim - 1));
		j = (int)floor(l_y*double(_ydim - 1));
		k = (int)floor(l_z*double(_zdim - 1));
	}

	__host__ __device__ void worldToUpperLeftIndex(double *itrans, const double w_x, const double w_y, const double w_z, int &i, int &j, int &k) const {
		double l_x, l_y, l_z;
		worldToLocal(itrans, w_x, w_y, w_z, l_x, l_y, l_z);
		localToUpperLeftIndex(l_x, l_y, l_z, i, j, k);
	}

	__host__ __device__ bool localIsValid(double l_x, double l_y, double l_z) const {
		return l_x >= 0.0 && l_x < 1.0 && l_y >= 0.0 && l_y < 1.0 && l_z >= 0.0 && l_z < 1.0;
	}

	__host__ __device__ bool worldIsValid(double *itrans, const double w_x, const double w_y, const double w_z) const {
		double l_x, l_y, l_z;
		worldToLocal(itrans, w_x, w_y, w_z, l_x, l_y, l_z);
		return localIsValid(l_x, l_y, l_z);
	}

	__host__ __device__ double valueAtWorld(double *g, double *itrans, double* trans, double w_x, double w_y, double w_z) const {

		int i, j, k;
		double w_x0, w_y0, w_z0;
		double w_x1, w_y1, w_z1;
		double x, y, z;

		worldToUpperLeftIndex(itrans, w_x, w_y, w_z, i, j, k);
		indexToWorld(trans, i, j, k, w_x0, w_y0, w_z0);
		indexToWorld(trans, i + 1, j + 1, k + 1, w_x1, w_y1, w_z1);

		x = (w_x - w_x0) / (w_x1 - w_x0);
		y = (w_y - w_y0) / (w_y1 - w_y0);
		z = (w_z - w_z0) / (w_z1 - w_z0);

		//return linearInterpolate(g, i, j, k, x, y, z);
		/* TODO: */
		return 1.0;
	}

	__host__ __device__ void setTransform(double *t) {
		for (int i = 0; i < 16; i++){
			_trans[i] = t[i];
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				_mat_trans[i][j] = _trans[i + j * 4];
			}
		}

		_mat_itrans = glm::inverse(_mat_trans);

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				_itrans[i + j * 4] = _mat_itrans[i][j];
			}
		}
	}

	__host__ __device__ int getSize() const { return _xdim*_ydim*_zdim; }

	__host__ __device__ double dx() const {
		if (_xdim != 0) {
			double x1, x0, y, z;
			indexToWorld(_trans, 0, 0, 0, x0, y, z);
			indexToWorld(_trans, 1, 0, 0, x1, y, z);
			return fabs(x1 - x0);
		}
		return 0;
	}

	__host__ __device__ double dy() const {
		if (_ydim != 0) {
			double x, y0, y1, z;
			indexToWorld(_trans, 0, 0, 0, x, y0, z);
			indexToWorld(_trans, 0, 1, 0, x, y1, z);
			return fabs(y1 - y0);
		}
		return 0;
	}

	__host__ __device__ double dz() const {
		if (_zdim != 0) {
			double x, y, z0, z1;
			indexToWorld(_trans, 0, 0, 0, x, y, z0);
			indexToWorld(_trans, 0, 0, 1, x, y, z1);
			return fabs(z1 - z0);
		}
		return 0;
	}
};

#endif
