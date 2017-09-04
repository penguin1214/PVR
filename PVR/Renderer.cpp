#include "Renderer.h"
#include <fstream>
#include <algorithm>

void Renderer::saveImage(float *image) {
	char filename[2048];
	sprintf(filename, "fire%04d.ppm", num_file++);
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << _x << " " << _y << "\n255\n";
	for (int i = 0; i < _x; i++)  // loop over pixels, write RGB values
		for (int j = 0; j < _y; j++) {
			int index = i*_x + j;
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 0]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 1]) * 255);
			ofs << (unsigned char)((std::min)(1.0f, image[index * 3 + 2]) * 255);

		}
	printf("Saved image \n");
}