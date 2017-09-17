#ifndef VECTOR3_H
#define VECTOR3_H
#include <iostream>

class Vector3 {
public:
	// Constructors
	__host__ __device__ Vector3();
	__host__ __device__ Vector3(double xf, double yf, double zf);
	__host__ __device__ Vector3(double v);
	__host__ __device__ ~Vector3();

	// Functions
	__host__ __device__ void mult(double f);
	__host__ __device__ void substract(Vector3 *v);
	__host__ __device__ void rotate(Vector3 *axisRotate, double *angle, double a, double b, double c);

	__host__ __device__ double length();
	__host__ __device__ void normalize();
	__host__ __device__ double dot(const Vector3 &vTemp);
	__host__ __device__ static double dot(const Vector3 &v1, const Vector3 &v2);

	__host__ __device__ double norm() const;
	__host__ __device__ double angle(Vector3 *vTemp);
	__host__ __device__ double cosA(Vector3 *vTemp);

	__host__ __device__ void translate(Vector3 *v);

	__host__ __device__ Vector3 cross(Vector3 *vTemp);
	__host__ __device__ Vector3 from(Vector3 *v);
	__host__ __device__ Vector3 add(Vector3 *v);
	__host__ __device__ Vector3 projectionOnVector(Vector3 v);

	//Overloaded operator
	__host__ __device__ void operator+= (const Vector3 &v);
	__host__ __device__ void operator-= (const Vector3 &v);
	__host__ __device__ void operator*= (const Vector3 &v);
	__host__ __device__ Vector3 operator* (const Vector3 &v) const;
	__host__ __device__ Vector3 operator+ (const Vector3 &v) const;
	__host__ __device__ Vector3 operator- (const Vector3 &v) const;
	__host__ __device__ bool operator==(const Vector3 &v) const;

	//__host__ __device__ void operator*= (const double f);
	__host__ __device__ void operator*= (double f);
	__host__ __device__ Vector3 operator * (const double f) const;
	__host__ __device__ Vector3 operator/ (const double f) const;
	__host__ __device__ Vector3 operator+ (const double f) const;
	__host__ __device__ Vector3 operator- (const double f) const;

	__host__ __device__ friend Vector3 operator+ (const double f, const Vector3 &v);
	__host__ __device__ friend Vector3 operator- (const double f, const Vector3 &v);
	__host__ __device__ friend Vector3 operator* (const double f, const Vector3 &v);

	__host__ __device__ Vector3 operator- () const;
	__host__ __device__ bool operator> (const Vector3 &v) const;

	friend std::ostream& operator<<(std::ostream& out, Vector3& v) {
		out << v.x << " " << v.y << " " << v.z << std::endl;
		return out;
	}

	__host__ __device__ void rotY(double angle); //rotera runt y axeln

												 //__host__ __device__ bool rayBoxIntersection(const Vector3 &minbox, const Vector3 &maxbox, const Vector3 &lineOrigin, const Vector3 &lineDirection, double *tmin, double *tmax);

	void description();

public:
	// Variables
	double x, y, z;
};
#endif