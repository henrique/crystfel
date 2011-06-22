/*
 * post-refinement.c
 *
 * Post refinement
 *
 * (c) 2006-2011 Thomas White <taw@physics.org>
 *
 * Part of CrystFEL - crystallography with a FEL
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <stdlib.h>
#include <assert.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

#include "image.h"
#include "post-refinement.h"
#include "peaks.h"
#include "symmetry.h"
#include "geometry.h"
#include "cell.h"


/* Maximum number of iterations of NLSq to do for each image per macrocycle. */
#define MAX_CYCLES (5)


/* Returns dp/dr at "r" */
static double partiality_gradient(double r, double profile_radius)
{
	double q, dpdq, dqdr;

	/* Calculate degree of penetration */
	q = (r + profile_radius)/(2.0*profile_radius);

	/* dp/dq */
	dpdq = 6.0*(q-pow(q, 2.0));

	/* dq/dr */
	dqdr = 1.0 / (2.0*profile_radius);

	return dpdq * dqdr;
}


/* Returns dp/drad at "r" */
static double partiality_rgradient(double r, double profile_radius)
{
	double q, dpdq, dqdrad;

	/* Calculate degree of penetration */
	q = (r + profile_radius)/(2.0*profile_radius);

	/* dp/dq */
	dpdq = 6.0*(q-pow(q, 2.0));

	/* dq/drad */
	dqdrad = 0.5 * (1.0 - r * pow(profile_radius, -2.0));

	return dpdq * dqdrad;
}


/* Return the gradient of parameter 'k' given the current status of 'image'. */
double gradient(struct image *image, int k, Reflection *refl, double r)
{
	double ds, tt, azix, aziy;
	double nom, den;
	double g;
	double asx, asy, asz;
	double bsx, bsy, bsz;
	double csx, csy, csz;
	double xl, yl, zl;
	signed int hs, ks, ls;
	double r1, r2, p;
	int clamp_low, clamp_high;

	get_symmetric_indices(refl, &hs, &ks, &ls);

	cell_get_reciprocal(image->indexed_cell, &asx, &asy, &asz,
	                                         &bsx, &bsy, &bsz,
	                                         &csx, &csy, &csz);
	xl = hs*asx + ks*bsx + ls*csx;
	yl = hs*asy + ks*bsy + ls*csy;
	zl = hs*asz + ks*bsz + ls*csz;

	ds = 2.0 * resolution(image->indexed_cell, hs, ks, ls);
	tt = angle_between(0.0, 0.0, 1.0, xl, yl, zl+1.0/image->lambda);
	azix = angle_between(1.0, 0.0, 0.0, xl, yl, 0.0);
	aziy = angle_between(0.0, 1.0, 0.0, xl, yl, 0.0);
	get_partial(refl, &r1, &r2, &p, &clamp_low, &clamp_high);

	/* Calculate the gradient of partiality wrt excitation error. */
	g = 0.0;
	if ( clamp_low == 0 ) {
		g -= partiality_gradient(r1, r);
	}
	if ( clamp_high == 0 ) {
		g += partiality_gradient(r2, r);
	}

	/* For many gradients, just multiply the above number by the gradient
	 * of excitation error wrt whatever. */
	switch ( k ) {

	case REF_DIV :
		nom = sqrt(2.0) * ds * sin(image->div/2.0);
		den = sqrt(1.0 - cos(image->div/2.0));
		return (nom/den) * g;

	case REF_R :
		g = 0.0;
		if ( clamp_low == 0 ) {
			g += partiality_rgradient(r1, r);
		}
		if ( clamp_high == 0 ) {
			g += partiality_rgradient(r2, r);
		}
		return g;

	/* Cell parameters and orientation */
	case REF_ASX :
		return hs * sin(tt) * cos(azix) * g;
	case REF_BSX :
		return ks * sin(tt) * cos(azix) * g;
	case REF_CSX :
		return ls * sin(tt) * cos(azix) * g;
	case REF_ASY :
		return hs * sin(tt) * cos(aziy) * g;
	case REF_BSY :
		return ks * sin(tt) * cos(aziy) * g;
	case REF_CSY :
		return ls * sin(tt) * cos(aziy) * g;
	case REF_ASZ :
		return hs * cos(tt) * g;
	case REF_BSZ :
		return ks * cos(tt) * g;
	case REF_CSZ :
		return ls * cos(tt) * g;

	}

	ERROR("No gradient defined for parameter %i\n", k);
	abort();
}


static void apply_cell_shift(UnitCell *cell, int k, double shift)
{
	double asx, asy, asz;
	double bsx, bsy, bsz;
	double csx, csy, csz;

	cell_get_reciprocal(cell, &asx, &asy, &asz,
	                          &bsx, &bsy, &bsz,
	                          &csx, &csy, &csz);

	switch ( k )
	{
		case REF_ASX :  asx += shift;  break;
		case REF_ASY :  asy += shift;  break;
		case REF_ASZ :  asz += shift;  break;
		case REF_BSX :  bsx += shift;  break;
		case REF_BSY :  bsy += shift;  break;
		case REF_BSZ :  bsz += shift;  break;
		case REF_CSX :  csx += shift;  break;
		case REF_CSY :  csy += shift;  break;
		case REF_CSZ :  csz += shift;  break;
	}

	cell_set_reciprocal(cell, asx, asy, asz,
	                          bsx, bsy, bsz,
	                          csx, csy, csz);
}


/* Apply the given shift to the 'k'th parameter of 'image'. */
static void apply_shift(struct image *image, int k, double shift)
{
	switch ( k ) {

	case REF_DIV :
		image->div += shift;
		break;

	case REF_R :
		image->profile_radius += shift;
		break;

	case REF_ASX :
	case REF_ASY :
	case REF_ASZ :
	case REF_BSX :
	case REF_BSY :
	case REF_BSZ :
	case REF_CSX :
	case REF_CSY :
	case REF_CSZ :
		apply_cell_shift(image->indexed_cell, k, shift);
		break;

	default :
		ERROR("No shift defined for parameter %i\n", k);
		abort();

	}
}


static void show_eigen(gsl_vector *e_val)
{
	int i;
	double vmax, vmin;
	const int n = e_val->size;

	STATUS("Eigenvalues:\n");
	vmin = +INFINITY;
	vmax = 0.0;
	for ( i=0; i<n; i++ ) {
		double val = gsl_vector_get(e_val, i);
		STATUS("%i: %e\n", i, val);
		if ( val > vmax ) vmax = val;
		if ( val < vmin ) vmin = val;
	}
	STATUS("Condition number: %e / %e = %e\n", vmin, vmax, vmin/vmax);
}


static gsl_vector *solve_svd(gsl_vector *v, gsl_matrix *M)
{
	gsl_matrix *s_vec;
	gsl_vector *s_val;
	int err, n;
	gsl_vector *shifts;

	n = v->size;
	if ( v->size != M->size1 ) return NULL;
	if ( v->size != M->size2 ) return NULL;

	s_val = gsl_vector_calloc(n);
	s_vec = gsl_matrix_calloc(n, n);

	err = gsl_linalg_SV_decomp_jacobi(M, s_vec, s_val);
	if ( err ) {
		ERROR("SVD failed: %s\n", gsl_strerror(err));
		gsl_matrix_free(s_vec);
		gsl_vector_free(s_val);
		return NULL;
	}
	/* "M" is now "U" */

	show_eigen(s_val);

	shifts = gsl_vector_calloc(n);
	err = gsl_linalg_SV_solve(M, s_vec, s_val, v, shifts);
	if ( err ) {
		ERROR("Matrix solution failed: %s\n", gsl_strerror(err));
		gsl_matrix_free(s_vec);
		gsl_vector_free(s_val);
		gsl_vector_free(shifts);
		return NULL;
	}

	gsl_matrix_free(s_vec);
	gsl_vector_free(s_val);

	return shifts;
}


static gsl_vector *solve_householder(gsl_vector *v, gsl_matrix *M)
{
	int n, err;
	gsl_vector *shifts;

	n = v->size;
	if ( v->size != M->size1 ) return NULL;
	if ( v->size != M->size2 ) return NULL;

	shifts = gsl_vector_alloc(n);

	err = gsl_linalg_HH_solve(M, v, shifts);

	if ( err != 0 ) {
		ERROR("Failed to solve equations: %s\n", gsl_strerror(err));
		gsl_vector_free(shifts);
		return NULL;
	}

	return shifts;
}


/* Perform one cycle of post refinement on 'image' against 'full' */
static double pr_iterate(struct image *image, const RefList *full,
                         const char *sym)
{
	gsl_matrix *M;
	gsl_vector *v;
	gsl_vector *shifts;
	int param;
	Reflection *refl;
	RefListIterator *iter;
	RefList *reflections;
	double max_shift;
	int nref = 0;

	reflections = image->reflections;

	M = gsl_matrix_calloc(NUM_PARAMS, NUM_PARAMS);
	v = gsl_vector_calloc(NUM_PARAMS);

	/* Construct the equations, one per reflection in this image */
	for ( refl = first_refl(reflections, &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) ) {

		signed int ha, ka, la;
		double I_full, delta_I;
		double I_partial;
		int k;
		double p;
		Reflection *match;
		double gradients[NUM_PARAMS];

		if ( !get_scalable(refl) ) continue;

		/* Find the full version */
		get_indices(refl, &ha, &ka, &la);
		match = find_refl(full, ha, ka, la);
		if ( match == NULL ) continue;
		/* Some reflections may have recently become scalable, but
		 * scale_intensities() might not yet have been called, so the
		 * full version may not have been calculated yet. */
		I_full = get_intensity(match);

		/* Actual measurement of this reflection from this pattern? */
		I_partial = get_intensity(refl);
		p = get_partiality(refl);
		delta_I = I_partial - (p * image->osf * I_full);

		/* Calculate all gradients for this reflection */
		for ( k=0; k<NUM_PARAMS; k++ ) {
			double gr;
			gr = gradient(image, k, refl, image->profile_radius);
			gradients[k] = gr;
		}

		for ( k=0; k<NUM_PARAMS; k++ ) {

			int g;
			double v_c, v_curr;
			double gr;

			for ( g=0; g<NUM_PARAMS; g++ ) {

				double M_c, M_curr;

				M_c = gradients[g] * gradients[k];
				M_c *= pow(image->osf * I_full, 2.0);

				M_curr = gsl_matrix_get(M, g, k);
				gsl_matrix_set(M, g, k, M_curr + M_c);

			}

			gr = gradients[k];
			v_c = delta_I * image->osf * I_full * gr;
			v_curr = gsl_vector_get(v, k);
			gsl_vector_set(v, k, v_curr + v_c);

		}

		nref++;

	}
	//show_matrix_eqn(M, v, NUM_PARAMS);

	STATUS("%i reflections were scalable\n", nref);
	if ( nref == 0 ) {
		ERROR("No reflections left to scale!\n");
		return 0.0;
	}

	max_shift = 0.0;
	//shifts = solve_householder(v, M);
	shifts = solve_svd(v, M);
	if ( shifts != NULL ) {

		for ( param=0; param<NUM_PARAMS; param++ ) {
			double shift = gsl_vector_get(shifts, param);
			apply_shift(image, param, shift);
			//STATUS("Shift %i: %e\n", param, shift);
			if ( fabs(shift) > max_shift ) max_shift = fabs(shift);
		}

	} else {
		ERROR("Problem solving equations.\n");
		/* Leave things as they were */
	}

	gsl_matrix_free(M);
	gsl_vector_free(v);
	gsl_vector_free(shifts);

	return max_shift;
}


static double mean_partial_dev(struct image *image,
                               const RefList *full, const char *sym)
{
	double dev = 0.0;

	/* For each reflection */
	Reflection *refl;
	RefListIterator *iter;

	for ( refl = first_refl(image->reflections, &iter);
	      refl != NULL;
	      refl = next_refl(refl, iter) ) {

		double G, p;
		signed int h, k, l;
		Reflection *full_version;
		double I_full, I_partial;

		if ( !get_scalable(refl) ) continue;

		get_indices(refl, &h, &k, &l);
		assert((h!=0) || (k!=0) || (l!=0));

		if ( !get_scalable(refl) ) continue;

		full_version = find_refl(full, h, k, l);
		if ( full_version == NULL ) continue;
		/* Some reflections may have recently become scalable, but
		 * scale_intensities() might not yet have been called, so the
		 * full version may not have been calculated yet. */

		G = image->osf;
		p = get_partiality(refl);
		I_partial = get_intensity(refl);
		I_full = get_intensity(full_version);
		//STATUS("%3i %3i %3i  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f\n",
		//       h, k, l, G, p, I_partial, I_full,
		//       I_partial - p*G*I_full);

		dev += pow(I_partial - p*G*I_full, 2.0);

	}

	return dev;
}


void pr_refine(struct image *image, const RefList *full, const char *sym)
{
	double max_shift, dev;
	int i;
	const int verbose = 1;
	int nexp, nfound, nnotfound;

	update_partialities(image, sym, NULL, &nexp, &nfound, &nnotfound);

	if ( verbose ) {
		dev = mean_partial_dev(image, full, sym);
		STATUS("PR starting dev = %5.2f (%i out of %i found)\n",
		       dev, nfound, nexp);
	}

	i = 0;
	do {

		double dev;

		max_shift = pr_iterate(image, full, sym);

		update_partialities(image, sym, NULL,
		                    &nexp, &nfound, &nnotfound);

		if ( verbose ) {
			dev = mean_partial_dev(image, full, sym);
			STATUS("PR Iteration %2i: max shift = %5.2f"
			       " dev = %5.2f (%i out of %i found)\n",
			       i+1, max_shift, dev, nfound, nexp);
		}

		i++;

	} while ( (max_shift > 0.01) && (i < MAX_CYCLES) );

	cell_print(image->indexed_cell);
}
