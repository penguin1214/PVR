#ifndef CAMERA_H
#define CAMERA_H

#include "Vector3.h"

class Film {
public:
	Film() {}
	Film(double wid, double hei) :_w(wid), _h(hei) {}

	double _w, _h;	// film size
	double _vfov;
	double _nearPlaneDistance;

	double aspectRatio() { return _w / _h; }
	void setFilm(double w, double h, double vfov, double dis) {
		_w = w; _h = h; _vfov = vfov; _nearPlaneDistance = dis;
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