#include "Renderer.h"
#include <fstream>
#include <algorithm>

#include "util.h"

void Renderer::saveImage(float *image) {
	char filename[2048];
	sprintf(filename, "smoke.ppm", num_file++);
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << _cam->_film->_w << " " << _cam->_film->_h << "\n255\n";
	for (int j = 0; j < _cam->_film->_h; j++)  // loop over pixels, write RGB values
		for (int i = 0; i < _cam->_film->_w; i++) {
			int index = j*_cam->_film->_w + i;
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 0]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 1]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 2]) * 255);

		}
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
