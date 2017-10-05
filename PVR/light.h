#ifndef PVR_LIGHT_H
#define PVR_LIGHT_H

#include "vector3.h"

class Light {
public:
	Vector3 pos;
	Vector3 color;
	Vector3 power;
	float intensity;
	Light() { pos = Vector3(-100, 100, -50); color = Vector3(1.0); intensity = 10.0; power = Vector3(1.0); }
};

class PointLight : public Light {
public:
	PointLight() { pos = Vector3(1.5, 1.5, 1.0); color = Vector3(1.0); intensity = 1.0; power = Vector3(100.0); }
};

#endif
