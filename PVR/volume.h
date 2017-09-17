#ifndef VOLUME_H
#define VOLUME_H

#include "grid.h"

class Volume {
public:
	Volume() {
		_grid = new Grid();
	}

	double transparency() {
		return exp(-_ot*_wds);
	}
	void setCoefficent(double oa, double ot, double os) {
		_oa = oa; _ot = ot; _os = os;
	}

	virtual double radiance(double lambda, double T) = 0;
	double _oa, _ot, _os;	// oa: absorption, ot: extinction(attennuation), os: scatter
	double _wds; // ray caster sample step
	Grid *_grid;
};


class BlackBody : public Volume {
public:
	BlackBody() {
		_grid = new Grid();
	}

	const double C_1 = 3.7418e-16;
	const double C_2 = 1.4388e-2;
	const double LeScale = 0.25;

	virtual double radiance(double lambda, double T) override {
		return (2.0*C_1) / (pow(lambda, 5.0) * (exp(C_2 / (lambda*T)) - 1.0));
	}
};

#endif