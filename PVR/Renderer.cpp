#include "Renderer.h"
#include <fstream>
#include <algorithm>

#include "util.h"
#include "core.h"

//////////////////////////////////////////////////////////////////////////
/// NON-MEMBER FUNC DECLARATION
//////////////////////////////////////////////////////////////////////////
Vector3 diffuse(Vector3 normal);

void Renderer::saveImage(float *image) {
	char filename[2048];
	sprintf(filename, "smoke.ppm", num_file++);
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << Presets::RESOLUTION_X << " " << Presets::RESOLUTION_Y << "\n255\n";
	for (int j = 0; j < Presets::RESOLUTION_Y; j++)  // loop over pixels, write RGB values
		for (int i = 0; i < Presets::RESOLUTION_X; i++) {
			int index = j*Presets::RESOLUTION_X + i;
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 0]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 1]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 2]) * 255);

		}
	ofs.flush();
	printf("Saved image \n");
}
bool Renderer::rayBBoxIntersection(Vector3 minbox, Vector3 maxbox, const Vector3 &rayOrigin, const Vector3 &rayDirection, float &tmin, float &tmax) {
	Vector3 dir = rayDirection;
	Vector3 dirfrac;
	dirfrac.x = 1.0f / dir.x;
	dirfrac.y = 1.0f / dir.y;
	dirfrac.z = 1.0f / dir.z;

	double xmin = (minbox.x - rayOrigin.x)*dirfrac.x;
	double xmax = (maxbox.x - rayOrigin.x)*dirfrac.x;
	double ymin = (minbox.y - rayOrigin.y)*dirfrac.y;
	double ymax = (maxbox.y - rayOrigin.y)*dirfrac.y;
	double zmin = (minbox.z - rayOrigin.z)*dirfrac.z;
	double zmax = (maxbox.z - rayOrigin.z)*dirfrac.z;

	binsort(&xmax, &xmin);
	binsort(&ymax, &ymin);
	binsort(&zmax, &zmin);

	tmin = d_max(d_max(xmin, ymin), zmin);
	tmax = d_min(d_min(xmax, ymax), zmax);

	if (tmax < 0) return false;
	if (tmin > tmax) return false;

	return true;
}
bool Renderer::intersect(Ray &r, HitRecord &rec, float tmin, float tmax) {
	// iterate through all shapes and call shape instances' intersect function
	for (std::vector<Shape* >::iterator it = _shapes.begin(); it != _shapes.end(); ++it) {
		HitRecord tmpRec;
		if ((*it)->intersect(r, tmin, tmax, tmpRec) && abs(tmpRec._t) < abs(rec._t)) {	// abs() needed?
			rec = tmpRec;
		}
	}
	return rec._is_intersect;
}

/* Trace photon and store, once. */
void Renderer::photonTrace(Ray& r, Vector3 power, int depth) {
	// check depth
	if (depth > G_MAX_DEPTH) return;
	++depth;

	// cast ray
	// test intersection
	HitRecord rec;
	rec._is_intersect = false;
	bool is_intersect = intersect(r, rec, G_TMIN, G_TMAX);
	// store
	if (is_intersect) {
		// compute filtered power
		// construct Photon structure
		photonMapper->storePhoton();
		// test reflection
		float roulette = (float)rand() / (float)RAND_MAX;

		// only consider diffuse reflection
		if (roulette >= 0.3 && roulette < 0.7) {	// diffuse reflection
			Ray r_reflect(rec._p, diffuse(rec._normal));
			photonTrace(r_reflect, power, depth);
		}
		else
			return;
	}
}

//////////////////////////////////////////////////////////////////////////
/// NON-MEMBER FUNC
//////////////////////////////////////////////////////////////////////////
Vector3 diffuse(Vector3 normal) {
	float dot;
	Vector3 res;
	do {
		res.x = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
		res.y = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
		res.z = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
		res.normalize();
		dot = res.dot(normal);
	} while (dot >= 0);
	return res;
}

