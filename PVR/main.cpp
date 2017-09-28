#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <cmath>

#include <omp.h>

#include "core.h"
#include "util.h"
#include "Renderer.h"
#include "volume.h"
#include "shape.h"
#include "qeKDTree.h"
#include "qePhotonMapper.h"


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
	return c;
}


int main() {
	/*===========================================Read File================================================*/
	/*===========================================Renderer Init================================================*/
	Renderer renderer;
	/*===========================================Set Grid================================================*/
	/*===========================================Set Volume================================================*/
	/*===========================================Set Camera================================================*/
	Vector3 lookAt(0.3, 0.5, 0.5); Vector3 eyepos(0.0); Vector3 up(0.0, 1.0, 0.0);
	renderer._cam = new Camera(lookAt, eyepos, up);

	const float fovy = M_PI*0.33;	// 60 deg
	const float dis = 0.5;
	const float aspr = (float)Presets::RESOLUTION_X / (float)Presets::RESOLUTION_Y;
	renderer._cam->setFovDis(fovy, dis, aspr);
	/*===========================================Light================================================*/
	PointLight *pl = new PointLight();
	pl->pos = Vector3(300.0, 400.0, -100.0);
	renderer._lights.push_back(pl);
	// global ambient
	//Vector3 g_ambient(0.0, 0.2, 0.2);
	/*=========================================== Objects ================================================*/
	Sphere *sphere1 = new Sphere(Vector3(0.0), 100.0);
	Material *mat1 = new Material(Vector3(1.0, 1.0, 1.0), NULL, NULL, NULL);

	Sphere *sphere2 = new Sphere(Vector3(-200.0, 100.0, -500.0), 100.0);
	Material *mat2 = new Material(Vector3(0.0, 0.0, 0.0), NULL, NULL, NULL);

	Plane *plane3 = new Plane(Vector3(0.0, 1.0, 0.0), 300.0);
	Material *_mat3 = new Material(Vector3(1.0, 1.0, 1.0), NULL, NULL, NULL);

	// add shapes
	renderer._shapes.push_back(sphere1); renderer._shapes.push_back(sphere2); renderer._shapes.push_back(plane3);
	/*=========================================== Photon Map ================================================*/
	// single light source now
	PointLight light = *renderer._lights[0];
	Vector3 p_power = light.power / (float)G_NUM_PHOTON * 100;

	// build global map
	int num_cnt = 0;
	while (num_cnt < G_NUM_PHOTON) {
		// emit photon
		// random directions in hemisphere
		float x, y, z;
		do {
			x = -1.0f + (float)rand() / ((float)RAND_MAX / 2.0f);
			y = -1.0f + (float)rand() / ((float)RAND_MAX / 2.0f);
			z = -1.0f + (float)rand() / ((float)RAND_MAX);
		} while (x*x + y*y + z*z > 1.0);
		Vector3 dir(x, y, z);
		dir.normalize();
		// photon tracing
		Ray r(light.pos, dir);
		renderer.photonTrace(r, p_power, 1, 0.0, 1000.0); // TODO
	}
	/*=========================================== Render ================================================*/
	float *image = new float[Presets::RESOLUTION_X * Presets::RESOLUTION_Y * 3];
	std::cout << "Start rendering..." << std::endl;

	renderer.saveImage(image);

	int t;
	std::cin >> t;

	return 0;
}


// volume rendering
//int main() {
//	/*===========================================Presets Initialize================================================*/
//	/*===========================================Read File================================================*/
//	int num_file = 0;
//	int grid_x, grid_y, grid_z;
//	std::string str_filename = "car_data0200.bin";
//	const char* filename = str_filename.c_str();
//
//	float *D = loadBinary<float>(filename, grid_x, grid_y, grid_z);	// 199, 399, 199
//	std::cout << "Read file " << filename << "  done." << std::endl;
//	/*===========================================Renderer Init================================================*/
//	Renderer renderer;
//	renderer.num_file = 0;
//	/*===========================================Set Grid================================================*/
//	int grid_size = grid_x * grid_y * grid_z;
//	Grid *grid = new Grid();
//	grid->_data = D;
//	grid->setSize(grid_x, grid_y, grid_z);
//	/*===========================================Set Volume================================================*/
//	renderer._volume = new BlackBody(grid);
//	renderer._volume->setCoefficent(2, 1, 1);
//	// bbox
//	Vector3 min_coord(0.0); Vector3 max_coord(grid_x - 1, grid_y - 1, grid_z - 1);
//	renderer._volume->setBox(min_coord, max_coord);
//	// material
//	Vector3 v_ambient(0.0, 0.1, 0.1);
//	Vector3 v_diffuse(0.2);
//	Vector3 v_specular(0.3);
//	Material *vmat = new Material(v_ambient, v_diffuse, v_specular, 32);
//	renderer._volume->_mat = vmat;
//	/*===========================================Set Camera================================================*/
//	Vector3 lookAt(0.3, 0.5, 0.5); Vector3 eyepos(0.3, 0.5, -0.6); Vector3 up(0.0, 1.0, 0.0);
//	renderer._cam = new Camera(lookAt, eyepos, up);
//
//	const float fovy = M_PI*0.33;	// 60 deg
//	const float dis = 0.5;
//	const float aspr = (float)Presets::RESOLUTION_X / (float)Presets::RESOLUTION_Y;
//
//	renderer._cam->setFovDis(fovy, dis, aspr);
//	/*===========================================Light================================================*/
//	PointLight *pl = new PointLight();
//	renderer._lights.push_back(pl);
//
//	// global ambient
//	Vector3 g_ambient(0.0, 0.2, 0.2);
//	/*=========================================== Sphere ================================================*/
//	Vector3 sphere_o(0.3, 0.7, 0.3);
//	float sphere_r = 0.05;
//	Sphere sphere(sphere_o, sphere_r);
//	// bind material
//	Vector3 sphere_ambient(0.3, 0.3, 0.0);
//	Vector3 sphere_diffuse(0.2);
//	Vector3 sphere_specular(0.8);
//	int sphere_shine = 32;
//	Material *mat = new Material(sphere_ambient, sphere_diffuse, sphere_specular, sphere_shine);
//	sphere._mat = mat;
//	/*=========================================== Render ================================================*/
//	float *image = new float[Presets::RESOLUTION_X * Presets::RESOLUTION_Y * 3];
//	std::cout << "Start rendering..." << std::endl;
//
//	// ray marching
//	float u, v;
//	float tmin, tmax;
//	float stride;
//	Vector3 sample_pos;
//	float density;
//	Vector3 col;
//
//	int xdim = grid_x; int ydim = grid_y; int zdim = grid_z;
//
//	Vector3 *gradient = renderer._volume->fetchGradient();
//
//#pragma omp parallel
//	{
//#pragma omp for
//		for (int y = 0; y < Presets::RESOLUTION_Y; y++) {
//			v = 0.5 - float(y) / float(Presets::RESOLUTION_Y);
//			std::cout << (float)y / float(Presets::RESOLUTION_Y) << std::endl;
//			for (int x = 0; x < Presets::RESOLUTION_X; x++) {
//				u = -0.5 + float(x) / float(Presets::RESOLUTION_X);
//
//				float T = 1.0;	// transparency
//				Vector3 Lo(0.0);
//
//				// pixel pos
//				Vector3 cursor = renderer._cam->_eyepos + renderer._cam->_forward*renderer._cam->_nearPlaneDistance + u*renderer._cam->_horizontal + v*renderer._cam->_vertical;
//				Vector3 ray_dir = cursor - renderer._cam->_eyepos;
//				ray_dir.normalize();
//
//				HitRecord rec;
//				bool has_surface = sphere.intersect(Ray(cursor, ray_dir), 0, 1000, rec);
//				has_surface = false;
//
//				// only intersect sphere
//				/*if (has_surface) {
//					Ray r_scnd(rec._p, renderer._lights[0]->pos - rec._p);
//					float temp_cos = r_scnd.direction.dot(rec._normal);
//					if (temp_cos > 0) {
//						Lo += temp_cos * renderer._lights[0]->color * sphere_color;
//					}
//				}*/
//
//				if (renderer.rayBBoxIntersection(renderer._volume->_min_coord, renderer._volume->_max_coord, cursor, ray_dir, tmin, tmax)) {
//
//					if (has_surface) tmax = rec._t;	// if has shape, update valid sample path
//
//					if (tmin > 0) {
//						stride = (tmax - tmin) / (Presets::NUM_RAY_SAMPLES + 1);
//
//						for (int n = 0; n < Presets::NUM_RAY_SAMPLES; n++) {
//							/* Use back-to-front compositing */
//							sample_pos = cursor + ray_dir*tmax - (n + 1)*stride*ray_dir;
//							float l_T = 1.0;
//
//							if (sample_pos.x > renderer._volume->_min_coord.x && sample_pos.x < renderer._volume->_max_coord.x
//								&& sample_pos.y > renderer._volume->_min_coord.y && sample_pos.y < renderer._volume->_max_coord.y
//								&& sample_pos.z > renderer._volume->_min_coord.z && sample_pos.z < renderer._volume->_max_coord.z) {
//
//								Vector3 sample_idx = renderer._volume->pos_map(sample_pos);
//								density = trilinearInterp(D, xdim, ydim, zdim, sample_idx.x, sample_idx.y, sample_idx.z);
//
//								if (density > 0) {
//									//T *= 1.0 - density * stride *renderer._volume->_oa;
//									T *= std::exp(-renderer._volume->_oa*density*stride);
//
//									//Vector3 light_ray = sample_pos - renderer._lights[0]->pos;
//									Vector3 light_ray = renderer._lights[0]->pos - sample_pos;
//									float l_stride = light_ray.length() / (Presets::NUM_LIGHT_RAY_SAMPLES + 1);
//									light_ray.normalize();
//
//									// check if blocked
//									//HitRecord tmp_rec;
//									//if (sphere.intersect(Ray(sample_pos, light_ray), 0, 1000, tmp_rec)) break;	// no light
//
//									for (int j = 0; j < Presets::NUM_LIGHT_RAY_SAMPLES; j++) {
//										// compute the radiance at the sample point
//										Vector3 l_sample_pos = sample_pos - j*l_stride*light_ray;
//										if (l_sample_pos.x > renderer._volume->_min_coord.x && l_sample_pos.x < renderer._volume->_max_coord.x
//											&& l_sample_pos.y > renderer._volume->_min_coord.y && l_sample_pos.y < renderer._volume->_max_coord.y
//											&& l_sample_pos.z > renderer._volume->_min_coord.z && l_sample_pos.z < renderer._volume->_max_coord.z) {
//
//											Vector3 l_sample_idx = renderer._volume->pos_map(l_sample_pos);
//											float l_density = trilinearInterp(D, xdim, ydim, zdim, l_sample_idx.x, l_sample_idx.y, l_sample_idx.z);
//											l_T *= std::exp(-renderer._volume->_oa*l_density*l_stride); // T = e^(-k(t)*dx)
//										}
//									}
//									Vector3 Li = renderer._lights[0]->intensity * (1.0 - l_T);	// light radiance at sampled position
//									// Blinn-Phong Shading
//									Vector3 R(0.0);
//									// ambient
//									R += renderer._volume->_mat->k_ambient * g_ambient;
//									// diffuse
//									Vector3 g = gradient[renderer._volume->_grid->gridIndexAt(sample_idx.x, sample_idx.y, sample_idx.z)];
//									R += renderer._volume->_mat->k_diffuse * Li * std::fmax(0, g.dot(light_ray));
//									// specular
//									Vector3 view_ray = Vector3(sample_pos - renderer._cam->_eyepos);
//									view_ray.normalize();
//									R += renderer._volume->_mat->k_specular * Li * pow(view_ray.dot(light_ray), renderer._volume->_mat->k_shine);
//
//									//Lo += Li*(1.0 - T);
//									Lo += R*(1.0 - T);
//								}
//							}
//						}
//
//						if (has_surface) {
//							Ray r_scnd(rec._p, renderer._lights[0]->pos - rec._p);
//							float temp_cos = r_scnd.direction.dot(rec._normal);
//							if (temp_cos > 0) {
//								Lo += temp_cos * renderer._lights[0]->color * sphere._mat->k_diffuse * sphere._mat->k_ambient;
//							}
//						}
//					}
//				}
//				else {
//					if (has_surface) {
//						Ray r_scnd(rec._p, renderer._lights[0]->pos - rec._p);
//						float temp_cos = r_scnd.direction.dot(rec._normal);
//						if (temp_cos > 0) {
//							Lo += temp_cos * renderer._lights[0]->color * sphere._mat->k_diffuse * sphere._mat->k_ambient;
//						}
//					}
//				}
//
//				image[3 * (y*(int)Presets::RESOLUTION_X + x) + 0] = Lo.x;
//				image[3 * (y*(int)Presets::RESOLUTION_X + x) + 1] = Lo.y;
//				image[3 * (y*(int)Presets::RESOLUTION_X + x) + 2] = Lo.z;
//			}
//		}
//	}
//
//	renderer.saveImage(image);
//
//	int t;
//	std::cin >> t;
//
//	return 0;
//}