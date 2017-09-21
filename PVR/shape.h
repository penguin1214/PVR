#ifndef PVR_SHAPE_H
#define PVR_SHAPE_H

#include "vector3.h"
#include "camera.h"
#include "material.h"

struct hit_record {
	float t;
	Vector3 p;
	Vector3 norm;
	//material* mat_ptr;
};


class Shape {
public:
	Shape() {}
};

class Sphere : public Shape {
public:
	Sphere() {}
	Sphere(Vector3 c, float r) : _center(c), _radius(r), _mat(nullptr) {}

	bool intersect(Ray &r, float t_min, float t_max, hit_record &rec) {
		Vector3 oc = r.origin - _center;
		float a = r.direction.dot(r.direction);
		float b = oc.dot(r.direction);
		float c = oc.dot(oc) - _radius*_radius;
		float disc = b*b - a*c;
		if ((sqrt(disc)) > 0) {
			// calculate hit point
			float temp = (-b - sqrt(b * b - a * c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.origin + r.direction*temp;
				rec.norm = rec.p - _center;
				rec.norm.normalize();
				return true;
			}
			temp = (-b + sqrt(b*b - a*c)) / a;
			if (temp < t_max && temp > t_min) {
				rec.t = temp;
				rec.p = r.origin + r.direction*temp;
				rec.norm = rec.p - _center;
				rec.norm.normalize();
				return true;
			}
		}
		return false;
	}

	Vector3 _center;
	float _radius;
	Material *_mat;
};

#endif
