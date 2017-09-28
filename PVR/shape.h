#ifndef PVR_SHAPE_H
#define PVR_SHAPE_H

#include "vector3.h"
#include "camera.h"
#include "material.h"

class Shape;

struct HitRecord {
	bool is_intersect;
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
		Vector3 oc = r._o - _center;
		float a = r._d.dot(r._d);
		float b = oc.dot(r._d);
		float c = oc.dot(oc) - _radius*_radius;
		float disc = b*b - a*c;
		if ((sqrt(disc)) > 0) {
			// calculate hit point
			float temp = (-b - sqrt(b * b - a * c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.is_intersect = true;
				rec._t = temp;
				rec._p = r._o + r._d*temp;
				rec._normal = rec._p - _center;
				rec._normal.normalize();
				return true;
			}
			temp = (-b + sqrt(b*b - a*c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.is_intersect = true;
				rec._t = temp;
				rec._p = r._o + r._d*temp;
				rec._normal = rec._p - _center;
				rec._normal.normalize();
				return true;
			}
		}
		return false;
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
		float denom = _normal.dot(r._d);
		if (denom != 0) {
			rec._t = -1 * (_normal.dot(r._o) + _d) / denom;
			rec.is_intersect = true;
			return true;
		}
		else {
			rec._t = -1;
			return false;
		}
	}
};

#endif
