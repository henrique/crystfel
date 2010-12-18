/*
 * index.c
 *
 * Perform indexing (somehow)
 *
 * (c) 2006-2010 Thomas White <taw@physics.org>
 * (c) 2010 Richard Kirian <rkirian@asu.edu>
 *
 * Part of CrystFEL - crystallography with a FEL
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "image.h"
#include "utils.h"
#include "peaks.h"
#include "dirax.h"
#include "mosflm.h"
#include "sfac.h"
#include "detector.h"
#include "index.h"
#include "index-priv.h"
#include "templates.h"


/* Base class constructor for unspecialised indexing private data */
static IndexingPrivate *indexing_private(IndexingMethod indm)
{
	struct _indexingprivate *priv;
	priv = calloc(1, sizeof(struct _indexingprivate));
	priv->indm = indm;
	return priv;
}


IndexingPrivate **prepare_indexing(IndexingMethod *indm, UnitCell *cell,
                                  const char *filename, struct detector *det,
                                  double nominal_photon_energy)
{
	int n;
	int nm = 0;
	IndexingPrivate **iprivs;

	while ( indm[nm] != INDEXING_NONE ) nm++;
	STATUS("Preparing %i indexing methods.\n", nm);
	iprivs = malloc((nm+1) * sizeof(IndexingPrivate *));

	for ( n=0; n<nm; n++ ) {

		switch ( indm[n] ) {
		case INDEXING_NONE :
			ERROR("Tried to prepare INDEXING_NONE!\n");
			break;
		case INDEXING_DIRAX :
			 iprivs[n] = indexing_private(indm[n]);
			 break;
		case INDEXING_MOSFLM :
			 iprivs[n] = indexing_private(indm[n]);
			 break;
		case INDEXING_TEMPLATE :
			iprivs[n] = generate_templates(cell, filename, det,
				                  nominal_photon_energy);
			break;
		}

		n++;

	}
	iprivs[n] = NULL;

	return iprivs;
}


void cleanup_indexing(IndexingPrivate **priv)
{
	int n = 0;

	while ( priv[n] != NULL ) {

		switch ( priv[n]->indm ) {
		case INDEXING_NONE :
			free(priv[n]);
			break;
		case INDEXING_DIRAX :
			free(priv[n]);
			break;
		case INDEXING_MOSFLM :
			free(priv[n]);
			break;
		case INDEXING_TEMPLATE :
			free_templates(priv[n]);
		}

		n++;

	}
}


void map_all_peaks(struct image *image)
{
	int i;

	/* Map positions to 3D */
	for ( i=0; i<image_feature_count(image->features); i++ ) {

		struct imagefeature *f;
		struct rvec r;

		f = image_get_feature(image->features, i);
		if ( f == NULL ) continue;

		r = get_q(image, f->x, f->y, 1, NULL, 1.0/image->lambda);
		f->rx = r.u;  f->ry = r.v;  f->rz = r.w;

	}
}


void index_pattern(struct image *image, UnitCell *cell, IndexingMethod *indm,
                   int cellr, int verbose, IndexingPrivate **ipriv)
{
	int i;
	int n = 0;

	map_all_peaks(image);

	while ( indm[n] != INDEXING_NONE ) {

		image->ncells = 0;

		/* Index as appropriate */
		switch ( indm[n] ) {
		case INDEXING_NONE :
			return;
		case INDEXING_DIRAX :
			STATUS("Running DirAx...\n");
			run_dirax(image);
			break;
		case INDEXING_MOSFLM :
			STATUS("Running MOSFLM...\n");
			run_mosflm(image, cell);
			break;
		case INDEXING_TEMPLATE :
			match_templates(image, ipriv[n]);
			break;
		}

		if ( image->ncells == 0 ) {
			STATUS("No candidate cells found.\n");
			return;
		}

		if ( (cellr == CELLR_NONE) || (indm[n] == INDEXING_TEMPLATE) ) {
			image->indexed_cell = image->candidate_cells[0];
			if ( verbose ) {
				STATUS("--------------------\n");
				STATUS("The indexed cell (matching not"
				       " performed):\n");
				cell_print(image->indexed_cell);
				STATUS("--------------------\n");
			}
			return;
		}

		for ( i=0; i<image->ncells; i++ ) {

			UnitCell *new_cell = NULL;

			if ( verbose ) {
				STATUS("--------------------\n");
				STATUS("Candidate cell %i (before matching):\n",
				       i);
				cell_print(image->candidate_cells[i]);
				STATUS("--------------------\n");
			}

			/* Match or reduce the cell as appropriate */
			switch ( cellr ) {
			case CELLR_NONE :
				/* Never happens */
				break;
			case CELLR_REDUCE :
				new_cell = match_cell(image->candidate_cells[i],
					              cell, verbose, 1);
				break;
			case CELLR_COMPARE :
				new_cell = match_cell(image->candidate_cells[i],
					              cell, verbose, 0);
				break;
			}

			image->indexed_cell = new_cell;
			if ( new_cell != NULL ) {
				STATUS("Matched on attempt %i.\n", i);
				goto done;
			}

		}

		/* Move on to the next indexing method */
		n++;

	}

done:
	for ( i=0; i<image->ncells; i++ ) {
		cell_free(image->candidate_cells[i]);
	}
}


IndexingMethod *build_indexer_list(const char *str, int *need_cell)
{
	int n, i;
	char **methods;
	IndexingMethod *list;
	*need_cell = 0;

	n = assplode(str, ",", &methods, ASSPLODE_NONE);
	list = malloc((n+1)*sizeof(IndexingMethod));

	for ( i=0; i<n; i++ ) {

		if ( strcmp(methods[i], "dirax") == 0) {
			list[i] = INDEXING_DIRAX;
		} else if ( strcmp(methods[i], "mosflm") == 0) {
			list[i] = INDEXING_MOSFLM;
		} else if ( strcmp(methods[i], "template") == 0) {
			list[i] = INDEXING_TEMPLATE;
			*need_cell = 1;
		} else {
			ERROR("Unrecognised indexing method '%s'\n",
			      methods[i]);
			return NULL;
		}

		free(methods[i]);

	}
	free(methods);
	list[i] = INDEXING_NONE;

	return list;
}
