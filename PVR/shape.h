#ifndef PVR_SHAPE_H
#define PVR_SHAPE_H

#include "vector3.h"
#include "camera.h"
#include "material.h"

class Shape;

struct HitRecord {
	bool _is_intersect;
	int _idx;	// object index in _shapes array;
	float _t;
	Vector3 _p;    // point coord
	Vector3 _normal;
	Shape* _obj;
};


class Shape {
public:
	Shape() {}

	Material *_mat;
	
	virtual bool intersect(Ray &r, float t_min, float t_max, HitRecord &rec) = 0;
};

class Sphere : public Shape {
public:
	Sphere() {}
	Sphere(Vector3 c, float r) : _center(c), _radius(r) {
		_mat = nullptr;
	}
	~Sphere() {}

	virtual bool intersect(Ray &r, float t_min, float t_max, HitRecord &rec) {
		rec._obj = this;

		Vector3 L = _center - r._o;
		float tca = L.dot(r._d);
		if (tca < 0) return false;

		float d2 = L.dot(L) - tca * tca;
		if (d2 > _radius*_radius) return false;
		float thc = sqrt(_radius*_radius - d2);
		float t0 = tca - thc; float t1 = tca + thc;

		if (t0 > t1) std::swap(t0, t1);
		if (t0 < 0) {
			t0 = t1;
			if (t0 < 0) return false;
		}
		rec._t = t0;
		rec._p = r._o + r._d * t0;
		rec._normal = rec._p - _center;
		rec._normal.normalize();
		return true;
	}

	Vector3 _center;
	float _radius;
	// Material *_mat;
};

class Plane : public Shape {
public:
	Plane(Vector3 n, float d) :_normal(n), _d(d) {
		_mat = nullptr;
	}
	~Plane() {}

	Vector3 _normal;
	float _d;	// len of edge
	// Material *_mat
	virtual bool intersect(Ray &r, float t_min, float t_max, HitRecord &rec) {
		rec._obj = this;
		float denom = _normal.dot(r._d);
		if (denom != 0) {
			rec._t = -1 * (_normal.dot(r._o) + _d) / denom;
			rec._p = r._o + rec._t * r._d;
			rec._normal = _normal;
			rec._is_intersect = true;
			return true;
		}
		else {
			rec._t = -1;
			return false;
		}
	}
};

#endif
