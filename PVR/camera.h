#ifndef CAMERA_H
#define CAMERA_H

#include "Vector3.h"

class Ray {
public:
	Vector3 origin;
	Vector3 direction;

	Ray(Vector3 o, Vector3 d) : origin(o) { d.normalize(); direction = d; }
};

class Film {
public:
	Film() {}
	Film(double wid, double hei) :_w(wid), _h(hei) {}

	int _w, _h;	// film size
	double _vfov;
	double _nearPlaneDistance;

	double aspectRatio() { return _w / _h; }
	void setFilm(double w, double h, double vfov, double dis) {
		_w = w; _h = h; _vfov = vfov; _nearPlaneDistance = dis;
	}
	void setFovAndDis(double fov, double dis, double aspr) {
		_vfov = fov; _nearPlaneDistance = dis;
		_h = 2 * _nearPlaneDistance*tan(fov / 2);
		_w = _h * aspr;
		std::cout << _w << ", " << _h << std::endl;
	}
	void setSize(int w, int h) {
		_w = w; _h = h;
	}
	void setFov(double fov) {
		_vfov = fov;
		_nearPlaneDistance = _h / (2 * tan(fov / 2));
		std::cout << "film vfov: " << _vfov << ", distance: " << _nearPlaneDistance << std::endl;
	}
	void setDis(double dis) {
		_nearPlaneDistance = dis;
		_vfov = 2 * atan(_h / (2 * _nearPlaneDistance));
		std::cout << "film vfov: " << _vfov << ", distance: " << _nearPlaneDistance << std::endl;
	}
};

class Camera {
public:
	Camera() {
		_film = new Film();
	}
	Camera(Vector3 eyep, Vector3 at, Vector3 r, Vector3 u, Vector3 f, double ang) :
		_eyepos(eyep), _look_at(at), _up(u), _right(r), _forward(f), _angle(ang) {
	}

	Vector3 _eyepos;
	Vector3 _look_at;
	Vector3 _up;
	Vector3 _right;
	Vector3 _forward;	// look_at - eyepos
	double _angle;
	Film *_film;
};

#endif