#ifndef QE_PHOTONMAPPER_H
#define QE_PHOTONMAPPER_H

class qePhoton {
public:
	qePhoton() {}
	~qePhoton() {}

	Vector3 pos;
	char p[4];	// power packed as 4 chars
	char phi, theta;	// compressed incident direction
	short flag;	// flag used in kdtree
};


class qePhotonMapper {
public:
	qePhotonMapper() {}
	~qePhotonMapper() {}

	void traceGlobalMap() {}
	void traceCausticsMap() {}
	void traceVolumeMap() {}

	void emitPhoton() {}
	void storePhoton() {
		std::cout << "photon stored." << std::endl;
	}
};

#endif
