#include "util.h"


double diffclock(clock_t clock1, clock_t clock2) {
	double diffticks = clock1 - clock2;
	double diffms = (diffticks) / (CLOCKS_PER_SEC / 1000);
	return diffms / 1000.0;
}