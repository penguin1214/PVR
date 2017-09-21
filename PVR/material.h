#ifndef PVR_MATERIAL_H
#define PVR_MATERIAL_H

#include "vector3.h"

class Material {
public:

	Material() {}
	Material(Vector3 a, Vector3 d, Vector3 s, int shine) : k_ambient(a), k_diffuse(d), k_specular(s), k_shine(shine) {}

	// coefficient in RGB
	Vector3 k_ambient;
	Vector3 k_diffuse;
	Vector3 k_specular;
	int k_shine;
};

#endif
