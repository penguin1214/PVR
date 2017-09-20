#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <cmath>

#include "core.h"
#include "util.h"
#include "Renderer.h"
#include "volume.h"

struct hit_record {
	float t;
	Vector3 p;
	Vector3 norm;
	//material* mat_ptr;
};


template<class T>
T* loadBinary(const char * fn, int &gridx, int &gridy, int &gridz) {
	if (sizeof(T) != sizeof(float))return nullptr;
	FILE *fp = fopen(fn, "rb");
	fread(&gridx, 1, sizeof(int), fp);
	fread(&gridy, 1, sizeof(int), fp);
	fread(&gridz, 1, sizeof(int), fp);

	//N = Nx, NN = Nx*Ny, total = Nx*Ny*Nz;
	//_MLCuMallocHost((void **)&data, total * sizeof(T));
	int total = gridx * gridy * gridz;
	T *data = new T[total];

	//fread(&data[0], sizeof(T), total, fp);
	fread(data, sizeof(T), total, fp);
	/*for (int i = 0; i < total; i++) {
		if (data[i] > 1) {
			std::cout << data[i] << ", ";
		}
	}*/
	fclose(fp);
	printf("loaded %s <%d,%d,%d>\n", fn, gridx, gridy, gridz);
	return data;
}

// i,j,k are floor()
int getIndex(int gridx, int gridy, int gridz, int i, int j, int k) {
	return k*gridx*gridy + j*gridx + i;
}

float trilinearInterp(float *data, int gridx, int gridy, int gridz, float x, float y, float z) {
	int i = floor(x); int j = floor(y); int k = floor(z);
	//std::cout << i << ", " << j << ", " << k << std::endl;
	float xd = (x - i); float yd = y - j; float zd = z - k;

	float c00 = data[getIndex(gridx, gridy, gridz, i, j, k)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j, k)] * xd;
	float c01 = data[getIndex(gridx, gridy, gridz, i, j, k + 1)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j, k + 1)] * xd;
	float c10 = data[getIndex(gridx, gridy, gridz, i, j + 1, k)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j + 1, k)] * xd;
	float c11 = data[getIndex(gridx, gridy, gridz, i, j + 1, k + 1)] * (1 - xd) + data[getIndex(gridx, gridy, gridz, i + 1, j + 1, k + 1)] * xd;

	float c0 = c00*(1 - yd) + c10*yd;
	float c1 = c01*(1 - yd) + c11*yd;

	float c = c0*(1 - zd) + c1*zd;
	if (c > 1) {
		std::cout << c << std::endl;
	}
	return c;
}

bool sphereHit(Vector3 center, float radius, Ray &r, float t_min, float t_max, hit_record &rec) {
	Vector3 oc = r.origin - center;
	float a = r.direction.dot(r.direction);
	float b = 2 * oc.dot(r.direction);
	float c = oc.dot(oc) - radius*radius;
	float disc = b*b - 4 * a*c;
	if ((sqrt(disc)) >= 0) {
		// calculate hit point
		float temp = (-b - sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.origin + r.direction*temp;
			rec.norm = rec.p - center;
			rec.norm.normalize();
			return true;
		}
	}
	return false;
}

int main() {
	/*===========================================Read File================================================*/
	int num_file = 0;
	int grid_x, grid_y, grid_z;
	std::string str_filename = "car_data0200.bin";
	const char* filename = str_filename.c_str();

	float *D = loadBinary<float>(filename, grid_x, grid_y, grid_z);	// 199, 399, 199
	std::cout << "Read file " << filename << "  done." << std::endl;
	/*===========================================Renderer Init================================================*/
	Renderer renderer;
	renderer.num_file = 0;
	/*===========================================Set Grid================================================*/
	int grid_size = grid_x * grid_y * grid_z;
	Grid *grid = new Grid();
	grid->_data = D;
	grid->setSize(grid_x, grid_y, grid_z);
	/*===========================================Set Volume================================================*/
	renderer._volume = new BlackBody(grid);
	renderer._volume->setCoefficent(3, 1, 1);
	// bbox
	Vector3 min_coord(0.0); Vector3 max_coord(grid_x-1, grid_y-1, grid_z-1);
	renderer._volume->setBox(min_coord, max_coord);
	/*===========================================Set Camera================================================*/
	Vector3 lookAt(0.5, 0.5, 0.5); Vector3 eyepos(0.5, 0.5, -1.0); Vector3 up(0.0, 1.0, 0.0);
	renderer._cam = new Camera(lookAt, eyepos, up);

	const float fovy = M_PI*0.3;	// 90 deg
	const float dis = 0.5;
	const float aspr = (float)Presets::RESOLUTION_X / (float)Presets::RESOLUTION_Y;

	renderer._cam->setFovDis(fovy, dis, aspr);
	/*===========================================Light================================================*/
	PointLight *pl = new PointLight();
	renderer._lights.push_back(pl);
	/*=========================================== Sphere ================================================*/
	Vector3 sphere_o(0.5, 0.5, 0.5);
	float sphere_r = 0.2;
	Vector3 sphere_color(0.3, 0.3, 0.0);
	/*=========================================== Render ================================================*/
	float *image = new float[Presets::RESOLUTION_X * Presets::RESOLUTION_Y * 3];

	// ray marching
	float u, v;
	float tmin, tmax;
	float stride;
	Vector3 sample_pos;
	float density;
	Vector3 col;

	int xdim = grid_x; int ydim = grid_y; int zdim = grid_z;

	for (int y = 0; y < Presets::RESOLUTION_Y; y++) {
		v = 0.5 - float(y) / float(Presets::RESOLUTION_Y);
		std::cout << (float)y / float(Presets::RESOLUTION_Y) << std::endl;
		for (int x = 0; x < Presets::RESOLUTION_X; x++) {
			u = -0.5 + float(x) / float(Presets::RESOLUTION_X);

			float T = 1.0;	// transparency
			Vector3 Lo(0.0);

			// pixel pos
			Vector3 cursor = renderer._cam->_eyepos + renderer._cam->_forward*renderer._cam->_nearPlaneDistance + u*renderer._cam->_horizontal + v*renderer._cam->_vertical;
			Vector3 ray_dir = cursor - renderer._cam->_eyepos;
			ray_dir.normalize();

			hit_record rec;
			bool has_surface = sphereHit(sphere_o, sphere_r, Ray(cursor, ray_dir), 0, 1000, rec);

			// only intersect sphere
			/*if (has_surface) {
				Ray r_scnd(rec.p, renderer._lights[0]->pos - rec.p);
				float temp_cos = r_scnd.direction.dot(rec.norm);
				std::cout << rec.p << std::endl;
				if (temp_cos > 0) {
					Lo += temp_cos * renderer._lights[0]->color * sphere_color;
				}
			}*/

			if (renderer.rayBBoxIntersection(renderer._volume->_min_coord, renderer._volume->_max_coord, cursor, ray_dir, tmin, tmax)) {

				//if (has_surface) tmax = rec.t;

				if (tmin > 0) {
					stride = (tmax - tmin) / (Presets::NUM_RAY_SAMPLES + 1);
#if 0

					// n = 0;
					sample_pos = cursor + ray_dir*tmax - stride*ray_dir;
					float l_T = 1.0;
					Vector3 light_ray = sample_pos - renderer._lights[0]->pos;
					float l_stride = light_ray.length() / (Presets::NUM_LIGHT_RAY_SAMPLES + 1);
					light_ray.normalize();

					for (int j = 0; j < Presets::NUM_LIGHT_RAY_SAMPLES; j++) {
						// compute the radiance at the sample point
						Vector3 l_sample_pos = sample_pos - j*l_stride*light_ray;
						if (l_sample_pos.x > 0 && l_sample_pos.x < grid_x && l_sample_pos.y > 0 && l_sample_pos.y < grid_y && l_sample_pos.z > 0 && l_sample_pos.z < grid_z) {
							int l_index = l_sample_pos.x + l_sample_pos.y*renderer._volume->_grid->_xdim + l_sample_pos.z*renderer._volume->_grid->_xdim*renderer._volume->_grid->_ydim;
							float l_density = trilinearInterp(D, xdim, ydim, zdim, l_sample_pos.x, l_sample_pos.y, l_sample_pos.z);
							l_T *= std::exp(-renderer._volume->_oa*l_density*l_stride); // T = e^(-k(t)*dx)
						}
					}

					if (sample_pos.x > 0 && sample_pos.x < xdim && sample_pos.y > 0 && sample_pos.y < ydim && sample_pos.z < xdim && sample_pos.z > 0) {
						density = trilinearInterp(D, xdim, ydim, zdim, sample_pos.x, sample_pos.y, sample_pos.z);

						if (density > 0) {
							//T *= 1.0 - density * stride *renderer._volume->_oa;
							T *= std::exp(-renderer._volume->_oa*density*stride);
						}
					}

					Vector3 Li(0.0);
					Li = renderer._lights[0]->color * (1.0 - l_T);	// light radiance at sampled position

					if (has_surface) {
						Ray r_scnd(rec.p, renderer._lights[0]->pos - rec.p);
						r_scnd.direction.normalize(); rec.norm.normalize();
						float temp_cos = r_scnd.direction.dot(rec.norm);
						if (temp_cos > 0) {
							Lo += temp_cos * Li * sphere_color;
						}
					}
					else {
						Lo += Li*(1.0 - T);
					}

#endif
					// n > 0
					for (int n = 0; n < Presets::NUM_RAY_SAMPLES; n++) {
						/* Use back-to-front compositing */
						sample_pos = cursor + ray_dir*tmax - (n + 1)*stride*ray_dir;
						float l_T = 1.0;

						if (sample_pos.x > renderer._volume->_min_coord.x && sample_pos.x < renderer._volume->_max_coord.x
							&& sample_pos.y > renderer._volume->_min_coord.y && sample_pos.y < renderer._volume->_max_coord.y
							&& sample_pos.z > renderer._volume->_min_coord.z && sample_pos.z < renderer._volume->_max_coord.z) {

							Vector3 sample_idx = renderer._volume->pos_map(sample_pos);
							density = trilinearInterp(D, xdim, ydim, zdim, sample_idx.x, sample_idx.y, sample_idx.z);

							float tmp_d = sqrt(pow(sample_pos.x - sphere_o.x, 2) + pow(sample_pos.y - sphere_o.y, 2) + pow(sample_pos.z - sphere_o.z, 2));
							if (tmp_d < sphere_r) break;

							if (density > 0) {
								//T *= 1.0 - density * stride *renderer._volume->_oa;
								T *= std::exp(-renderer._volume->_oa*density*stride);

								Vector3 light_ray = sample_pos - renderer._lights[0]->pos;
								float l_stride = light_ray.length() / (Presets::NUM_LIGHT_RAY_SAMPLES + 1);
								light_ray.normalize();

								for (int j = 0; j < Presets::NUM_LIGHT_RAY_SAMPLES; j++) {
									// compute the radiance at the sample point
									Vector3 l_sample_pos = sample_pos - j*l_stride*light_ray;
									if (l_sample_pos.x > renderer._volume->_min_coord.x && l_sample_pos.x < renderer._volume->_max_coord.x
										&& l_sample_pos.y > renderer._volume->_min_coord.y && l_sample_pos.y < renderer._volume->_max_coord.y
										&& l_sample_pos.z > renderer._volume->_min_coord.z && l_sample_pos.z < renderer._volume->_max_coord.z) {

										Vector3 l_sample_idx = renderer._volume->pos_map(l_sample_pos);
										float l_density = trilinearInterp(D, xdim, ydim, zdim, l_sample_idx.x, l_sample_idx.y, l_sample_idx.z);
										l_T *= std::exp(-renderer._volume->_oa*l_density*l_stride); // T = e^(-k(t)*dx)
									}
								}
								Vector3 Li = renderer._lights[0]->intensity * (1.0 - l_T);	// light radiance at sampled position
								Lo += Li*(1.0 - T);
							}
						}
					}

					if (has_surface) {
						Ray r_scnd(rec.p, renderer._lights[0]->pos - rec.p);
						float temp_cos = r_scnd.direction.dot(rec.norm);
						if (temp_cos > 0) {
							Lo += temp_cos * renderer._lights[0]->color * sphere_color;
						}
					}
				}
			}
			else {
				if (has_surface) {
					Ray r_scnd(rec.p, renderer._lights[0]->pos - rec.p);
					float temp_cos = r_scnd.direction.dot(rec.norm);
					if (temp_cos > 0) {
						Lo += temp_cos * renderer._lights[0]->color * sphere_color;
					}
				}
			}

			image[3 * (y*(int)Presets::RESOLUTION_X + x) + 0] = Lo.x;
			image[3 * (y*(int)Presets::RESOLUTION_X + x) + 1] = Lo.y;
			image[3 * (y*(int)Presets::RESOLUTION_X + x) + 2] = Lo.z;
		}
	}
	/*for (int z = 0; z < grid_z; z++) {
		for (int y = 0; y < grid_y; y++) {
			for (int x = 0; x < grid_x; x++) {
			}
		}
	}*/

	std::cout << "Start rendering..." << std::endl;
	renderer.saveImage(image);

	int t;
	std::cin >> t;

	return 0;
}