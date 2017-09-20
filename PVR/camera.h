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
};

class Camera {
public:
	Camera() {
		_film = new Film();
	}
	Camera(Vector3 eyep, Vector3 at, Vector3 r, Vector3 u, Vector3 f, double ang) :
		_eyepos(eyep), _look_at(at), _up(u), _right(r), _forward(f) {
		_film = new Film();
	}
	Camera(Vector3 lookat, Vector3 eyep, Vector3 up) : _look_at(lookat), _eyepos(eyep){
		_film = new Film();
		// rotate camera according to angle
		/*eyepos -= lookAt;
		eyepos.rotY(6.28 * angle);
		eyepos += lookAt;*/
		_forward = _look_at - _eyepos;
		_forward.normalize();
		_right = up.cross(&_forward);
		_right.normalize();
		_up = _forward.cross(&_right);
		_up.normalize();

		/*std::cout << "lookat: " << lookAt << std::endl;
		std::cout << "eyepos: " << eyepos << std::endl;
		std::cout << "up: " << up << std::endl;
		std::cout << "forward: " << forward << std::endl;
		std::cout << "right : " << right << std::endl;*/
	}

	inline void setFovDis(float fov, float dis, float aspr) {
		_vfov = fov;
		_nearPlaneDistance = dis;
		_aspect_ratio = aspr;
		// update film size
		float height = 2 * dis*tan(0.5 * fov); float width = height * aspr;
		_horizontal = _right * width;
		_vertical = _up * height;
	}


	Vector3 _eyepos;
	Vector3 _look_at;
	Vector3 _up;
	Vector3 _right;
	Vector3 _forward;	// look_at - eyepos
	Vector3 _horizontal;	// horizontal length, i.e film width
	Vector3 _vertical;	// vertical length, i.e film height
	float _vfov;	// field of view
	float _nearPlaneDistance;	// distance between image plane and camera
	float _aspect_ratio;
	Film *_film;
};

#endif