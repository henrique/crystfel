/*
 * pinkindexer.c
 *
 * Interface to PinkIndexer
 *
 * Copyright © 2017-2019 Deutsches Elektronen-Synchrotron DESY,
 *                       a research centre of the Helmholtz Association.
 *
 * Authors:
 *   2017-2019 Yaroslav Gevorkov <yaroslav.gevorkov@desy.de>
 *
 * This file is part of CrystFEL.
 *
 * CrystFEL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CrystFEL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CrystFEL.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "pinkindexer.h"

#ifdef HAVE_PINKINDEXER

#include <stdlib.h>

#include "utils.h"
#include "cell-utils.h"
#include "peaks.h"

#include "pinkIndexer/adaptions/crystfel/Lattice.h"
#include "pinkIndexer/adaptions/crystfel/ExperimentSettings.h"
#include "pinkIndexer/adaptions/crystfel/PinkIndexer.h"

#define MAX_MULTI_LATTICE_COUNT 8

struct pinkIndexer_private_data {
	PinkIndexer *pinkIndexer;
	reciprocalPeaks_1_per_A_t reciprocalPeaks_1_per_A;
	float *intensities;

	IndexingMethod indm;
	UnitCell *cellTemplate;
	int threadCount;
	int multi;
	int min_peaks;

	int no_check_indexed;

	IntegerMatrix *centeringTransformation;
	LatticeTransform_t latticeReductionTransform;
};

//static void reduceCell(UnitCell* cell, LatticeTransform_t* appliedReductionTransform);
//static void restoreCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform);
static void reduceReciprocalCell(UnitCell* cell, LatticeTransform_t* appliedReductionTransform);
static void restoreReciprocalCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform);
static void makeRightHanded(UnitCell* cell);
static void update_detector(struct detector *det, double xoffs, double yoffs);

int run_pinkIndexer(struct image *image, void *ipriv)
{
	struct pinkIndexer_private_data* pinkIndexer_private_data = (struct pinkIndexer_private_data*) ipriv;
	reciprocalPeaks_1_per_A_t* reciprocalPeaks_1_per_A = &(pinkIndexer_private_data->reciprocalPeaks_1_per_A);
	float *intensities = pinkIndexer_private_data->intensities;

	int peakCountMax = image_feature_count(image->features);
	if (peakCountMax < 5) {
		int goodLatticesCount = 0;
		return goodLatticesCount;
	}
	reciprocalPeaks_1_per_A->peakCount = 0;
	for (int i = 0; i < peakCountMax && i < MAX_PEAK_COUNT_FOR_INDEXER; i++) {
		struct imagefeature *f;
		f = image_get_feature(image->features, i);
		if (f == NULL) {
			continue;
		}

		reciprocalPeaks_1_per_A->coordinates_x[reciprocalPeaks_1_per_A->peakCount] = f->rz * 1e-10;
		reciprocalPeaks_1_per_A->coordinates_y[reciprocalPeaks_1_per_A->peakCount] = f->rx * 1e-10;
		reciprocalPeaks_1_per_A->coordinates_z[reciprocalPeaks_1_per_A->peakCount] = f->ry * 1e-10;
		intensities[reciprocalPeaks_1_per_A->peakCount] = (float) (f->intensity);
		reciprocalPeaks_1_per_A->peakCount++;
	}
	int indexed = 0;
	Lattice_t indexedLattice[MAX_MULTI_LATTICE_COUNT];
	float center_shift[MAX_MULTI_LATTICE_COUNT][2];

	float maxRefinementDisbalance = 0.4;

	do {
		int peakCount = reciprocalPeaks_1_per_A->peakCount;
		int matchedPeaksCount = PinkIndexer_indexPattern(pinkIndexer_private_data->pinkIndexer,
		        &(indexedLattice[indexed]), center_shift[indexed], reciprocalPeaks_1_per_A, intensities,
		        maxRefinementDisbalance,
		        pinkIndexer_private_data->threadCount);

		if ((matchedPeaksCount >= 25 && matchedPeaksCount >= peakCount * 0.30)
		        || matchedPeaksCount >= peakCount * 0.4
		        || matchedPeaksCount >= 70
		        || pinkIndexer_private_data->no_check_indexed == 1)
		                {
			UnitCell *uc;
			uc = cell_new();

			Lattice_t *l = &(indexedLattice[indexed]);

			cell_set_reciprocal(uc, l->ay * 1e10, l->az * 1e10, l->ax * 1e10,
			        l->by * 1e10, l->bz * 1e10, l->bx * 1e10,
			        l->cy * 1e10, l->cz * 1e10, l->cx * 1e10);

			restoreReciprocalCell(uc, &pinkIndexer_private_data->latticeReductionTransform);

			UnitCell *new_cell_trans = cell_transform_intmat(uc, pinkIndexer_private_data->centeringTransformation);
			cell_free(uc);
			uc = new_cell_trans;

			cell_set_lattice_type(new_cell_trans, cell_get_lattice_type(pinkIndexer_private_data->cellTemplate));
			cell_set_centering(new_cell_trans, cell_get_centering(pinkIndexer_private_data->cellTemplate));
			cell_set_unique_axis(new_cell_trans, cell_get_unique_axis(pinkIndexer_private_data->cellTemplate));

			if (validate_cell(uc)) {
				ERROR("pinkIndexer: problem with returned cell!\n");
			}

			Crystal * cr = crystal_new();
			if (cr == NULL) {
				ERROR("Failed to allocate crystal.\n");
				return 0;
			}
			crystal_set_cell(cr, uc);
			crystal_set_det_shift(cr, center_shift[indexed][0], center_shift[indexed][1]);
			update_detector(image->det, center_shift[indexed][0], center_shift[indexed][1]);
			image_add_crystal(image, cr);
			indexed++;

		} else {
			break;
		}
	} while (pinkIndexer_private_data->multi
	        && indexed <= MAX_MULTI_LATTICE_COUNT
	        && reciprocalPeaks_1_per_A->peakCount >= pinkIndexer_private_data->min_peaks);

	return indexed;
}

void *pinkIndexer_prepare(IndexingMethod *indm, UnitCell *cell,
        struct pinkIndexer_options *pinkIndexer_opts)
{
	if (pinkIndexer_opts->beamEnergy == 0.0) {
		ERROR("For pinkIndexer, the photon_energy must be defined as a "
		      "constant in the geometry file\n");
		return NULL;
	}
	if (pinkIndexer_opts->beamBandwidth == 0.0) {
		STATUS("Using default bandwidth of 0.01 for pinkIndexer\n");
		pinkIndexer_opts->beamBandwidth = 0.01;
	}
	if (pinkIndexer_opts->detectorDistance == 0.0 && pinkIndexer_opts->refinement_type ==
	        REFINEMENT_TYPE_firstFixedThenVariableLatticeParametersCenterAdjustmentMultiSeed) {
		ERROR("Using center refinement makes it necessary to have the detector distance fixed in the geometry file!");
	}

	if(pinkIndexer_opts->detectorDistance <= 0.0){
		pinkIndexer_opts->detectorDistance = 0.25; //fake value
	}

	struct pinkIndexer_private_data* pinkIndexer_private_data = malloc(sizeof(struct pinkIndexer_private_data));
	allocReciprocalPeaks(&(pinkIndexer_private_data->reciprocalPeaks_1_per_A));
	pinkIndexer_private_data->intensities = malloc(MAX_PEAK_COUNT_FOR_INDEXER * sizeof(float));
	pinkIndexer_private_data->indm = *indm;
	pinkIndexer_private_data->cellTemplate = cell;
	pinkIndexer_private_data->threadCount = pinkIndexer_opts->thread_count;
	pinkIndexer_private_data->multi = pinkIndexer_opts->multi;
	pinkIndexer_private_data->min_peaks = pinkIndexer_opts->min_peaks;
	pinkIndexer_private_data->no_check_indexed = pinkIndexer_opts->no_check_indexed;

	UnitCell* primitiveCell = uncenter_cell(cell, &pinkIndexer_private_data->centeringTransformation, NULL);

	//reduceCell(primitiveCell, &pinkIndexer_private_data->latticeReductionTransform);
	reduceReciprocalCell(primitiveCell, &pinkIndexer_private_data->latticeReductionTransform);

	double asx, asy, asz, bsx, bsy, bsz, csx, csy, csz;
	int ret = cell_get_reciprocal(primitiveCell, &asx, &asy, &asz, &bsx, &bsy, &bsz, &csx, &csy, &csz);
	if (ret != 0) {
		ERROR("cell_get_reciprocal did not finish properly!");
	}

	Lattice_t lattice = { .ax = asz * 1e-10, .ay = asx * 1e-10, .az = asy * 1e-10,
	        .bx = bsz * 1e-10, .by = bsx * 1e-10, .bz = bsy * 1e-10,
	        .cx = csz * 1e-10, .cy = csx * 1e-10, .cz = csy * 1e-10 };

	float detectorDistance_m = pinkIndexer_opts->detectorDistance;
	float beamEenergy_eV = pinkIndexer_opts->beamEnergy;
	float nonMonochromaticity = pinkIndexer_opts->beamBandwidth;
	float reflectionRadius_1_per_A;
	if (pinkIndexer_opts->reflectionRadius < 0) {
		reflectionRadius_1_per_A = 0.02
		        * sqrt(lattice.ax * lattice.ax + lattice.ay * lattice.ay + lattice.az * lattice.az);
	}
	else {
		reflectionRadius_1_per_A = pinkIndexer_opts->reflectionRadius;
	}

	float divergenceAngle_deg = 0.01; //fake

	float tolerance = pinkIndexer_opts->tolerance;
	Lattice_t sampleReciprocalLattice_1_per_A = lattice;
	float detectorRadius_m = 0.03; //fake, only for prediction
	ExperimentSettings* experimentSettings = ExperimentSettings_new(beamEenergy_eV, detectorDistance_m,
	        detectorRadius_m, divergenceAngle_deg, nonMonochromaticity, sampleReciprocalLattice_1_per_A, tolerance,
	        reflectionRadius_1_per_A);

	consideredPeaksCount_t consideredPeaksCount = pinkIndexer_opts->considered_peaks_count;
	angleResolution_t angleResolution = pinkIndexer_opts->angle_resolution;
	refinementType_t refinementType = pinkIndexer_opts->refinement_type;
	float maxResolutionForIndexing_1_per_A = pinkIndexer_opts->maxResolutionForIndexing_1_per_A;
	pinkIndexer_private_data->pinkIndexer = PinkIndexer_new(experimentSettings, consideredPeaksCount, angleResolution,
	        refinementType,
	        maxResolutionForIndexing_1_per_A);

	ExperimentSettings_delete(experimentSettings);
	cell_free(primitiveCell);

	/* Flags that pinkIndexer knows about */
	*indm &= INDEXING_METHOD_MASK
	        | INDEXING_USE_CELL_PARAMETERS;

	return pinkIndexer_private_data;
}

//static void reduceCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform)
//{
//	double ax, ay, az, bx, by, bz, cx, cy, cz;
//	cell_get_cartesian(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);
//
//	Lattice_t l = { ax, ay, az, bx, by, bz, cx, cy, cz };
//
//	reduceLattice(&l, appliedReductionTransform);
//
//	cell_set_cartesian(cell, l.ax, l.ay, l.az,
//	        l.bx, l.by, l.bz,
//	        l.cx, l.cy, l.cz);
//
//	makeRightHanded(cell);
//}
//
//static void restoreCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform)
//{
//
//	double ax, ay, az, bx, by, bz, cx, cy, cz;
//	cell_get_cartesian(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);
//
//	Lattice_t l = { ax, ay, az, bx, by, bz, cx, cy, cz };
//
//	restoreLattice(&l, appliedReductionTransform);
//
//	cell_set_cartesian(cell, l.ax, l.ay, l.az,
//	        l.bx, l.by, l.bz,
//	        l.cx, l.cy, l.cz);
//
//	makeRightHanded(cell);
//}

static void reduceReciprocalCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform)
{
	double ax, ay, az, bx, by, bz, cx, cy, cz;
	cell_get_reciprocal(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);

	Lattice_t l = { ax, ay, az, bx, by, bz, cx, cy, cz };

	reduceLattice(&l, appliedReductionTransform);

	cell_set_reciprocal(cell, l.ax, l.ay, l.az,
	        l.bx, l.by, l.bz,
	        l.cx, l.cy, l.cz);

	makeRightHanded(cell);
}

static void restoreReciprocalCell(UnitCell *cell, LatticeTransform_t* appliedReductionTransform)
{

	double ax, ay, az, bx, by, bz, cx, cy, cz;
	cell_get_reciprocal(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);

	Lattice_t l = { ax, ay, az, bx, by, bz, cx, cy, cz };

	restoreLattice(&l, appliedReductionTransform);

	cell_set_reciprocal(cell, l.ax, l.ay, l.az,
	        l.bx, l.by, l.bz,
	        l.cx, l.cy, l.cz);

	makeRightHanded(cell);
}

static void makeRightHanded(UnitCell *cell)
{
	double ax, ay, az, bx, by, bz, cx, cy, cz;
	cell_get_cartesian(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);

	if (!right_handed(cell)) {
		cell_set_cartesian(cell, -ax, -ay, -az, -bx, -by, -bz, -cx, -cy, -cz);
	}
}

//hack for electron crystallography while crystal_set_det_shift is not working approprietly
static void update_detector(struct detector *det, double xoffs, double yoffs)
{
	int i;

	for (i = 0; i < det->n_panels; i++) {
		struct panel *p = &det->panels[i];
		p->cnx += xoffs * p->res;
		p->cny += yoffs * p->res;
	}
}

void pinkIndexer_cleanup(void *pp)
{
	struct pinkIndexer_private_data* pinkIndexer_private_data = (struct pinkIndexer_private_data*) pp;

	freeReciprocalPeaks(pinkIndexer_private_data->reciprocalPeaks_1_per_A);
	free(pinkIndexer_private_data->intensities);
	intmat_free(pinkIndexer_private_data->centeringTransformation);
	PinkIndexer_delete(pinkIndexer_private_data->pinkIndexer);
}

const char *pinkIndexer_probe(UnitCell *cell)
{
	return "pinkIndexer";
}

#else /* HAVE_PINKINDEXER */

int run_pinkIndexer(struct image *image, void *ipriv)
{
	ERROR("This copy of CrystFEL was compiled without PINKINDEXER support.\n");
	return 0;
}

extern void *pinkIndexer_prepare(IndexingMethod *indm, UnitCell *cell,
		struct pinkIndexer_options *pinkIndexer_opts)
{
	ERROR("This copy of CrystFEL was compiled without PINKINDEXER support.\n");
	ERROR("To use PINKINDEXER indexing, recompile with PINKINDEXER.\n");
	return NULL;
}

void pinkIndexer_cleanup(void *pp)
{
}

const char *pinkIndexer_probe(UnitCell *cell)
{
	return NULL;
}

#endif /* HAVE_PINKINDEXER */
