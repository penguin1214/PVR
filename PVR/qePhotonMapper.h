#ifndef QE_PHOTONMAPPER_H
#define QE_PHOTONMAPPER_H

#include <algorithm>
#include "shape.h"

class qePhoton {
public:
	qePhoton() {}
	qePhoton(Vector3 power, Vector3 pos, Vector3 dir) {
		_p_r = power.x; _p_g = power.y; _p_b = power.z;
		_pos = pos; _dir = dir;
	}
	~qePhoton() {}

	Vector3 _pos;
	//char p[4];	// power packed as 4 chars
	float _p_r, _p_g, _p_b;	// power in rgb band
	// phi = 256*atan2(dy,dx) / (2*	PI)
	// theta = 256*acos(dx) / PI
	//char phi, theta;	// compressed incident direction
	Vector3 _dir;
	short _flag;	// flag used in kdtree, 2 bits
};


class qePhotonMapper {
public:
	qePhotonMapper() {
		_nGlobalPhoton[0] = 0; _nGlobalPhoton[1] = 0; _nGlobalPhoton[2] = 0;
	}
	~qePhotonMapper() {}

	void traceGlobalMap() {}
	void traceCausticsMap() {}
	void traceVolumeMap() {}

	void emitPhoton() {}
	void storePhoton(int obj_idx, qePhoton p) {
		_global[obj_idx][_nGlobalPhoton[obj_idx]] = p;
		_nGlobalPhoton[obj_idx]++;
		//std::cout << "photon stored." << std::endl;
	}

	Vector3 lookUpGlobalMap(HitRecord &rec, float r) {
		Vector3 g_color;
		Vector3 pos = rec._p;
		int obj_idx = rec._idx;

		for (int i = 0; i < _nGlobalPhoton[obj_idx]; i++) {
			// for every photon stored for a particular object
			// test distance, use fixed radius instead of number n
			Vector3 p_pos = _global[obj_idx][i]._pos;
			float d2 = (pos.x - p_pos.x)*(pos.x - p_pos.x) + (pos.y - p_pos.y)*(pos.y - p_pos.y) + (pos.z - p_pos.z)*(pos.z - p_pos.z)
				- r*r;
			if (d2 < 0.0) {
				Vector3 p_dir = _global[obj_idx][i]._dir;
				float dis = (pos - p_pos).length();
				// TODO: negative?
				//float weight = std::max(0.0, -rec._normal.dot(p_dir));
				//weight *= (r - dis) / r;
				Vector3 p_power(_global[obj_idx][i]._p_r, _global[obj_idx][i]._p_g, _global[obj_idx][i]._p_b);
				//g_color += weight*p_power;
				g_color += p_power;
			}
		}
		return g_color;
	}

private:
	qePhoton _global[3][2000000];
	int _nGlobalPhoton[3];	// store number of photons for every object
};

#endif
