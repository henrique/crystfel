/*
 * indexamajig.c
 *
 * Index patterns, output hkl+intensity etc.
 *
 * Copyright © 2012 Deutsches Elektronen-Synchrotron DESY,
 *                  a research centre of the Helmholtz Association.
 * Copyright © 2012 Richard Kirian
 * Copyright © 2012 Lorenzo Galli
 *
 * Authors:
 *   2010-2012 Thomas White <taw@physics.org>
 *   2011      Richard Kirian
 *   2012      Lorenzo Galli
 *   2012      Chunhong Yoon
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


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <hdf5.h>
#include <gsl/gsl_errno.h>
#include <pthread.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>

#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#else
#include <sys/time.h>
#endif

#include "utils.h"
#include "hdf5-file.h"
#include "index.h"
#include "peaks.h"
#include "detector.h"
#include "filters.h"
#include "thread-pool.h"
#include "beam-parameters.h"
#include "geometry.h"
#include "stream.h"
#include "reflist-utils.h"


/* Write statistics at APPROXIMATELY this interval */
#define STATS_EVERY_N_SECONDS (5)

enum {
	PEAK_ZAEF,
	PEAK_HDF5,
};


/* Information about the indexing process which is common to all patterns */
struct index_args
{
	UnitCell *cell;
	int config_cmfilter;
	int config_noisefilter;
	int config_verbose;
	int stream_flags;         /* What goes into the output? */
	int config_satcorr;
	int config_closer;
	int config_insane;
	int config_bgsub;
	float threshold;
	float min_gradient;
	float min_snr;
	double min_int_snr;
	struct detector *det;
	IndexingMethod *indm;
	IndexingPrivate **ipriv;
	int peaks;                /* Peak detection method */
	int cellr;
	float tols[4];
	struct beam_params *beam;
	char *element;
	char *hdf5_peak_path;
	double ir_inn;
	double ir_mid;
	double ir_out;
	struct copy_hdf5_field *copyme;
};


/* Information about the indexing process for one pattern */
struct pattern_args
{
	/* "Input" */
	char *filename;

	/* "Output" */
	int indexable;
};


static void show_help(const char *s)
{
	printf("Syntax: %s [options]\n\n", s);
	printf(
"Process and index FEL diffraction images.\n"
"\n"
" -h, --help               Display this help message.\n"
"\n"
" -i, --input=<filename>   Specify file containing list of images to process.\n"
"                           '-' means stdin, which is the default.\n"
" -o, --output=<filename>  Write output stream to this file. '-' for stdout.\n"
"                           Default: indexamajig.stream\n"
"\n"
"     --indexing=<methods> Use 'methods' for indexing.  Provide one or more\n"
"                           methods separated by commas.  Choose from:\n"
"                            none     : no indexing (default)\n"
"                            dirax    : invoke DirAx\n"
"                            mosflm   : invoke MOSFLM (DPS)\n"
"                            reax     : DPS algorithm with known unit cell\n"
" -g. --geometry=<file>    Get detector geometry from file.\n"
" -b, --beam=<file>        Get beam parameters from file (provides nominal\n"
"                           wavelength value if no per-shot value is found in\n"
"                           the HDF5 files.\n"
" -p, --pdb=<file>         PDB file from which to get the unit cell to match.\n"
"                           Default: 'molecule.pdb'.\n"
"     --basename           Remove the directory parts of the filenames.\n"
" -x, --prefix=<p>         Prefix filenames from input file with <p>.\n"
"     --peaks=<method>     Use 'method' for finding peaks.  Choose from:\n"
"                           zaef  : Use Zaefferer (2000) gradient detection.\n"
"                                    This is the default method.\n"
"                           hdf5  : Get from a table in HDF5 file.\n"
"     --hdf5-peaks=<p>     Find peaks table in HDF5 file here.\n"
"                           Default: /processing/hitfinder/peakinfo\n"
"\n\n"
"You can control what information is included in the output stream using\n"
"' --record=<flag1>,<flag2>,<flag3>' and so on.  Possible flags are:\n\n"
" integrated        Include a list of reflection intensities, produced by\n"
"                    integrating around predicted peak locations.\n"
"\n"
" peaks             Include peak locations and intensities from the peak\n"
"                    search.\n"
"\n"
" peaksifindexed    As 'peaks', but only if the pattern could be indexed.\n"
"\n"
" peaksifnotindexed As 'peaks', but only if the pattern could NOT be indexed.\n"
"\n\n"
"The default is '--record=integrated'.\n"
"\n\n"
"For more control over the process, you might need:\n\n"
"  --cell-reduction=<m>  Use <m> as the cell reduction method. Choose from:\n"
"                         none    : no matching, just use the raw cell.\n"
"                         reduce  : full cell reduction.\n"
"                         compare : match by at most changing the order of\n"
"                                   the indices.\n"
"                         compare_ab : compare 'a' and 'b' lengths only.\n"
"    --tolerance=<tol>   Set the tolerances for cell reduction.\n"
"                          Default: 5,5,5,1.5.\n"
"    --filter-cm         Perform common-mode noise subtraction on images\n"
"                         before proceeding.  Intensities will be extracted\n"
"                         from the image as it is after this processing.\n"
"    --filter-noise      Apply an aggressive noise filter which sets all\n"
"                         pixels in each 3x3 region to zero if any of them\n"
"                         have negative values.  Intensity measurement will\n"
"                         be performed on the image as it was before this.\n"
"    --no-sat-corr       Don't correct values of saturated peaks using a\n"
"                         table included in the HDF5 file.\n"
"    --threshold=<n>     Only accept peaks above <n> ADU.  Default: 800.\n"
"    --min-gradient=<n>  Minimum gradient for Zaefferer peak search.\n"
"                         Default: 100,000.\n"
"    --min-snr=<n>       Minimum signal-to-noise ratio for peaks.\n"
"                         Default: 5.\n"
"    --min-integration-snr=<n> Minimum signal-to-noise ratio for peaks\n"
"                         during integration. Default: -infinity.\n"
"    --int-radius=<r>    Set the integration radii.  Default: 4,5,7.\n"
"-e, --image=<element>   Use this image from the HDF5 file.\n"
"                          Example: /data/data0.\n"
"                          Default: The first one found.\n"
"\n"
"\nFor time-resolved stuff, you might want to use:\n\n"
"     --copy-hdf5-field <f>  Copy the value of field <f> into the stream. You\n"
"                             can use this option as many times as you need.\n"
"\n"
"\nOptions for greater performance or verbosity:\n\n"
"     --verbose            Be verbose about indexing.\n"
" -j <n>                   Run <n> analyses in parallel.  Default 1.\n"
"\n"
"\nOptions you probably won't need:\n\n"
"     --no-check-prefix    Don't attempt to correct the --prefix.\n"
"     --closer-peak        Don't integrate from the location of a nearby peak\n"
"                           instead of the predicted spot.  Don't use.\n"
"     --insane             Don't check that the reduced cell accounts for at\n"
"                           least 10%% of the located peaks.\n"
"     --no-bg-sub          Don't subtract local background estimates from\n"
"                           integrated intensities.\n"
);
}


static char *get_pattern(FILE *fh, char **use_this_one_instead,
                         int config_basename, const char *prefix)
{
	char *line;
	char *filename;

	do {

		/* Get the next filename */
		if ( *use_this_one_instead != NULL ) {

			line = *use_this_one_instead;
			*use_this_one_instead = NULL;

		} else {

			char *rval;

			line = malloc(1024*sizeof(char));
			rval = fgets(line, 1023, fh);
			if ( rval == NULL ) {
				free(line);
				return NULL;
			}

		}

		chomp(line);

	} while ( strlen(line) == 0 );

	if ( config_basename ) {
		char *tmp;
		tmp = safe_basename(line);
		free(line);
		line = tmp;
	}

	filename = malloc(strlen(prefix)+strlen(line)+1);

	snprintf(filename, 1023, "%s%s", prefix, line);

	free(line);

	return filename;
}


static void process_image(const struct index_args *iargs,
                          struct pattern_args *pargs, FILE *ofh,
                          int cookie)
{
	float *data_for_measurement;
	size_t data_size;
	UnitCell *cell = iargs->cell;
	int config_cmfilter = iargs->config_cmfilter;
	int config_noisefilter = iargs->config_noisefilter;
	int config_verbose = iargs->config_verbose;
	IndexingMethod *indm = iargs->indm;
	struct beam_params *beam = iargs->beam;
	int r, check;
	struct hdfile *hdfile;
	struct image image;

	image.features = NULL;
	image.data = NULL;
	image.flags = NULL;
	image.indexed_cell = NULL;
	image.det = copy_geom(iargs->det);
	image.copyme = iargs->copyme;
	image.reflections = NULL;
	image.id = cookie;
	image.filename = pargs->filename;
	image.beam = beam;

	hdfile = hdfile_open(image.filename);
	if ( hdfile == NULL ) return;

	r = hdfile_set_first_image(hdfile, "/");
	if ( r ) {
		ERROR("Couldn't select first path\n");
		hdfile_close(hdfile);
		return;
	}

	check = hdf5_read(hdfile, &image, 1);
	if ( check ) {
		hdfile_close(hdfile);
		return;
	}

	if ( (image.width != image.det->max_fs + 1 )
	  || (image.height != image.det->max_ss + 1))
	{
		ERROR("Image size doesn't match geometry size"
			" - rejecting image.\n");
		ERROR("Image size: %i,%i.  Geometry size: %i,%i\n",
		image.width, image.height,
		image.det->max_fs + 1, image.det->max_ss + 1);
		hdfile_close(hdfile);
		free_detector_geometry(image.det);
		return;
	}

	if ( image.lambda < 0.0 ) {
		if ( beam != NULL ) {
			ERROR("Using nominal photon energy of %.2f eV\n",
			beam->photon_energy);
			image.lambda = ph_en_to_lambda(
			eV_to_J(beam->photon_energy));
		} else {
			ERROR("No wavelength in file, so you need to give "
				"a beam parameters file with -b.\n");
			hdfile_close(hdfile);
			free_detector_geometry(image.det);
			return;
		}
	}
	fill_in_values(image.det, hdfile);

	if ( config_cmfilter ) {
		filter_cm(&image);
	}

	/* Take snapshot of image after CM subtraction but before
	 * the aggressive noise filter. */
	data_size = image.width * image.height * sizeof(float);
	data_for_measurement = malloc(data_size);

	if ( config_noisefilter ) {
		filter_noise(&image, data_for_measurement);
	} else {
		memcpy(data_for_measurement, image.data, data_size);
	}

	switch ( iargs->peaks ) {

		case PEAK_HDF5:
		// Get peaks from HDF5
		if (get_peaks(&image, hdfile,
			iargs->hdf5_peak_path)) {
			ERROR("Failed to get peaks from HDF5 file.\n");
		}
		break;

		case PEAK_ZAEF:
		search_peaks(&image, iargs->threshold,
		             iargs->min_gradient, iargs->min_snr,
		             iargs->ir_inn, iargs->ir_mid, iargs->ir_out);
		break;

	}

	/* Get rid of noise-filtered version at this point
	 * - it was strictly for the purposes of peak detection. */
	free(image.data);
	image.data = data_for_measurement;

	/* Calculate orientation matrix (by magic) */
	image.div = beam->divergence;
	image.bw = beam->bandwidth;
	image.profile_radius = 0.0001e9;

	index_pattern(&image, cell, indm, iargs->cellr,
	              config_verbose, iargs->ipriv,
	              iargs->config_insane, iargs->tols);

	if ( image.indexed_cell != NULL ) {
		pargs->indexable = 1;
		image.reflections = find_intersections(&image,
				image.indexed_cell);
		if (image.reflections != NULL) {
			integrate_reflections(&image,
			                      iargs->config_closer,
			                      iargs->config_bgsub,
			                      iargs->min_int_snr,
			                      iargs->ir_inn,
			                      iargs->ir_mid,
			                      iargs->ir_out);
		}
	} else {
		image.reflections = NULL;
	}

	write_chunk(ofh, &image, hdfile, iargs->stream_flags);
	fprintf(ofh, "END\n");
	fflush(ofh);

	/* Only free cell if found */
	cell_free(image.indexed_cell);

	reflist_free(image.reflections);
	free(image.data);
	if ( image.flags != NULL ) free(image.flags);
	image_feature_list_free(image.features);
	hdfile_close(hdfile);
	free_detector_geometry(image.det);
}


static void run_work(const struct index_args *iargs,
                     int filename_pipe, int results_pipe, FILE *ofh,
                     int cookie)
{
	int allDone = 0;
	FILE *fh;

	fh = fdopen(filename_pipe, "r");
	if ( fh == NULL ) {
		ERROR("Failed to fdopen() the filename pipe!\n");
		close(filename_pipe);
		close(results_pipe);
		return;
	}

	while ( !allDone ) {

		struct pattern_args pargs;
		int  w, c;
		char buf[1024];
		char *line;
		char *rval;

		line = malloc(1024*sizeof(char));
		rval = fgets(line, 1023, fh);
		if ( rval == NULL ) {
			free(line);
			if ( feof(fh) ) {
				allDone = 1;
				continue;
			} else {
				ERROR("Read error!\n");
				return;
			}
		}

		chomp(line);

		if ( strlen(line) == 0 ) {

			allDone = 1;

		} else {

			pargs.filename = line;
			pargs.indexable = 0;

			process_image(iargs, &pargs, ofh, cookie);

			/* Request another image */
			c = sprintf(buf, "%i\n", pargs.indexable);
			w = write(results_pipe, buf, c);
			if ( w < 0 ) {
				ERROR("write P0\n");
			}

		}

		free(line);

	}

	/* close my pipes */
	fclose(fh);
	close(results_pipe);

	cleanup_indexing(iargs->ipriv);
	free(iargs->indm);
	free(iargs->ipriv);
	free_detector_geometry(iargs->det);
	free(iargs->beam);
	free(iargs->element);
	free(iargs->hdf5_peak_path);
	free_copy_hdf5_field_list(iargs->copyme);
	cell_free(iargs->cell);
}


#ifdef HAVE_CLOCK_GETTIME

static time_t get_monotonic_seconds()
{
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return tp.tv_sec;
}

#else

/* Fallback version of the above.  The time according to gettimeofday() is not
 * monotonic, so measuring intervals based on it will screw up if there's a
 * timezone change (e.g. daylight savings) while the program is running. */
static time_t get_monotonic_seconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec;
}

#endif


static int parse_cell_reduction(const char *scellr, int *err,
                                int *reduction_needs_cell)
{
	*err = 0;
	if ( strcmp(scellr, "none") == 0 ) {
		*reduction_needs_cell = 0;
		return CELLR_NONE;
	} else if ( strcmp(scellr, "reduce") == 0) {
		*reduction_needs_cell = 1;
		return CELLR_REDUCE;
	} else if ( strcmp(scellr, "compare") == 0) {
		*reduction_needs_cell = 1;
		return CELLR_COMPARE;
	} else if ( strcmp(scellr, "compare_ab") == 0) {
		*reduction_needs_cell = 1;
		return CELLR_COMPARE_AB;
	} else {
		*err = 1;
		*reduction_needs_cell = 0;
		return CELLR_NONE;
	}
}


static void pump_chunk(FILE *fh, int *finished, FILE *ofh)
{
	int chunk_started = 0;
	int chunk_finished = 0;

	do {

		char line[1024];
		char *rval;

		rval = fgets(line, 1024, fh);
		if ( rval == NULL ) {

			if ( feof(fh) ) {
				/* Process died */
				*finished = 1;
				if ( chunk_started ) {
					ERROR("EOF during chunk!\n");
					fprintf(ofh, "Chunk is unfinished!\n");
				}
			} else {
				ERROR("fgets() failed: %s\n", strerror(errno));
			}
			chunk_finished = 1;
			continue;

		}

		if ( strcmp(line, "END\n") == 0 ) {
			chunk_finished = 1;
		} else {
			chunk_started = 1;
			fprintf(ofh, "%s", line);
		}

	} while ( !chunk_finished );
}


static void run_reader(int *stream_pipe_read, int n_proc, FILE *ofh)
{
	int done = 0;
	int *finished;
	FILE **fhs;
	int i;

	finished = calloc(n_proc, sizeof(int));
	if ( finished == NULL ) {
		ERROR("Couldn't allocate memory for flags!\n");
		exit(1);
	}

	fhs = calloc(n_proc, sizeof(FILE *));
	if ( fhs == NULL ) {
		ERROR("Couldn't allocate memory for file handles!\n");
		exit(1);
	}

	for ( i=0; i<n_proc; i++ ) {
		fhs[i] = fdopen(stream_pipe_read[i], "r");
		if ( fhs[i] == NULL ) {
			ERROR("Couldn't fdopen() stream!\n");
			exit(1);
		}
	}

	while ( !done ) {

		int r, i;
		struct timeval tv;
		fd_set fds;
		int fdmax;

		tv.tv_sec = 5;
		tv.tv_usec = 0;

		FD_ZERO(&fds);
		fdmax = 0;
		for ( i=0; i<n_proc; i++ ) {

			int fd;

			if ( finished[i] ) continue;

			fd = stream_pipe_read[i];

			FD_SET(fd, &fds);
			if ( fd > fdmax ) fdmax = fd;

		}

		r = select(fdmax+1, &fds, NULL, NULL, &tv);

		if ( r == -1 ) {
			ERROR("select() failed: %s\n", strerror(errno));
			continue;
		}

		if ( r == 0 ) continue; /* Nothing this time.  Try again */

		for ( i=0; i<n_proc; i++ ) {

			if ( finished[i] ) continue;

			if ( !FD_ISSET(stream_pipe_read[i], &fds) ) continue;

			pump_chunk(fhs[i], &finished[i], ofh);

		}

		done = 1;
		for ( i=0; i<n_proc; i++ ) {
			if ( !finished[i] ) done = 0;
		}

	}

	free(finished);

	for ( i=0; i<n_proc; i++ ) {
		fclose(fhs[i]);
	}
	free(fhs);

	if ( ofh != stdout ) fclose(ofh);
}


static void signal_handler(int sig, siginfo_t *si, void *uc_v)
{
	struct ucontext_t *uc = uc_v;

	STATUS("Signal!\n");
}


int main(int argc, char *argv[])
{
	int c;
	char *filename = NULL;
	char *outfile = NULL;
	FILE *fh;
	FILE *ofh;
	char *rval = NULL;
	int config_noindex = 0;
	int config_cmfilter = 0;
	int config_noisefilter = 0;
	int config_verbose = 0;
	int config_satcorr = 1;
	int config_checkprefix = 1;
	int config_closer = 0;
	int config_insane = 0;
	int config_bgsub = 1;
	int config_basename = 0;
	float threshold = 800.0;
	float min_gradient = 100000.0;
	float min_snr = 5.0;
	double min_int_snr = -INFINITY;
	struct detector *det;
	char *geometry = NULL;
	IndexingMethod *indm;
	IndexingPrivate **ipriv;
	int indexer_needs_cell;
	int reduction_needs_cell;
	char *indm_str = NULL;
	UnitCell *cell;
	char *pdb = NULL;
	char *prefix = NULL;
	char *speaks = NULL;
	char *scellr = NULL;
	char *toler = NULL;
	float tols[4] = {5.0, 5.0, 5.0, 1.5}; /* a,b,c,angles (%,%,%,deg) */
	int cellr;
	int peaks;
	int n_proc = 1;
	char *prepare_line;
	char prepare_filename[1024];
	char *use_this_one_instead;
	struct index_args iargs;
	struct beam_params *beam = NULL;
	char *element = NULL;
	double nominal_photon_energy;
	int stream_flags = STREAM_INTEGRATED;
	char *hdf5_peak_path = NULL;
	struct copy_hdf5_field *copyme;
	char *intrad = NULL;
	float ir_inn = 4.0;
	float ir_mid = 5.0;
	float ir_out = 7.0;
	int n_indexable, n_processed, n_indexable_last_stats;
	int n_processed_last_stats;
	int t_last_stats;
	pid_t *pids;
	int *filename_pipes;
	int *stream_pipe_read;
	int *stream_pipe_write;
	FILE **result_fhs;
	int i;
	int allDone;
	int *finished;
	pid_t pr;
	struct sigaction sa;
	int r;

	copyme = new_copy_hdf5_field_list();
	if ( copyme == NULL ) {
		ERROR("Couldn't allocate HDF5 field list.\n");
		return 1;
	}

	/* Long options */
	const struct option longopts[] = {
		{"help",               0, NULL,               'h'},
		{"input",              1, NULL,               'i'},
		{"output",             1, NULL,               'o'},
		{"no-index",           0, &config_noindex,     1},
		{"indexing",           1, NULL,               'z'},
		{"geometry",           1, NULL,               'g'},
		{"beam",               1, NULL,               'b'},
		{"filter-cm",          0, &config_cmfilter,    1},
		{"filter-noise",       0, &config_noisefilter, 1},
		{"verbose",            0, &config_verbose,     1},
		{"pdb",                1, NULL,               'p'},
		{"prefix",             1, NULL,               'x'},
		{"no-sat-corr",        0, &config_satcorr,     0},
		{"sat-corr",           0, &config_satcorr,     1}, /* Compat */
		{"threshold",          1, NULL,               't'},
		{"no-check-prefix",    0, &config_checkprefix, 0},
		{"no-closer-peak",     0, &config_closer,      0},
		{"closer-peak",        0, &config_closer,      1},
		{"insane",             0, &config_insane,      1},
		{"image",              1, NULL,               'e'},
		{"basename",           0, &config_basename,    1},
		{"bg-sub",             0, &config_bgsub,       1}, /* Compat */
		{"no-bg-sub",          0, &config_bgsub,       0},

		{"peaks",              1, NULL,                2},
		{"cell-reduction",     1, NULL,                3},
		{"min-gradient",       1, NULL,                4},
		{"record",             1, NULL,                5},
		{"cpus",               1, NULL,                6},
		{"cpugroup",           1, NULL,                7},
		{"cpuoffset",          1, NULL,                8},
		{"hdf5-peaks",         1, NULL,                9},
		{"copy-hdf5-field",    1, NULL,               10},
		{"min-snr",            1, NULL,               11},
		{"min-integration-snr",1, NULL,               12},
		{"tolerance",          1, NULL,               13},
		{"int-radius",         1, NULL,               14},
		{0, 0, NULL, 0}
	};

	/* Short options */
	while ((c = getopt_long(argc, argv, "hi:o:z:p:x:j:g:t:b:e:",
	                        longopts, NULL)) != -1)
	{
		switch (c) {

			case 'h' :
			show_help(argv[0]);
			return 0;

			case 'i' :
			filename = strdup(optarg);
			break;

			case 'o' :
			outfile = strdup(optarg);
			break;

			case 'z' :
			indm_str = strdup(optarg);
			break;

			case 'p' :
			pdb = strdup(optarg);
			break;

			case 'x' :
			prefix = strdup(optarg);
			break;

			case 'j' :
			n_proc = atoi(optarg);
			break;

			case 'g' :
			geometry = strdup(optarg);
			break;

			case 't' :
			threshold = strtof(optarg, NULL);
			break;

			case 'b' :
			beam = get_beam_parameters(optarg);
			if ( beam == NULL ) {
				ERROR("Failed to load beam parameters"
				      " from '%s'\n", optarg);
				return 1;
			}
			break;

			case 'e' :
			element = strdup(optarg);
			break;

			case 2 :
			speaks = strdup(optarg);
			break;

			case 3 :
			scellr = strdup(optarg);
			break;

			case 4 :
			min_gradient = strtof(optarg, NULL);
			break;

			case 5 :
			stream_flags = parse_stream_flags(optarg);
			if ( stream_flags < 0 ) return 1;
			break;

			case 6 :
			case 7 :
			case 8 :
			ERROR("The options --cpus, --cpugroup and --cpuoffset"
			      " are no longer used by indexamajig.\n");
			break;

			case 9 :
			hdf5_peak_path = strdup(optarg);
			break;

			case 10 :
			add_copy_hdf5_field(copyme, optarg);
			break;

			case 11 :
			min_snr = strtof(optarg, NULL);
			break;

			case 12 :
			min_int_snr = strtof(optarg, NULL);
			break;

			case 13 :
			toler = strdup(optarg);
			break;

			case 14 :
			intrad = strdup(optarg);
			break;

			case 0 :
			break;

			default :
			return 1;

		}

	}

	if ( filename == NULL ) {
		filename = strdup("-");
	}
	if ( strcmp(filename, "-") == 0 ) {
		fh = stdin;
	} else {
		fh = fopen(filename, "r");
	}
	if ( fh == NULL ) {
		ERROR("Failed to open input file '%s'\n", filename);
		return 1;
	}
	free(filename);

	if ( outfile == NULL ) {
		ofh = stdout;
	} else {
		ofh = fopen(outfile, "w");
		if ( ofh == NULL ) {
			ERROR("Failed to open output file '%s'\n", outfile);
			return 1;
		}
		free(outfile);
	}

	if ( hdf5_peak_path == NULL ) {
		hdf5_peak_path = strdup("/processing/hitfinder/peakinfo");
	}

	if ( speaks == NULL ) {
		speaks = strdup("zaef");
		STATUS("You didn't specify a peak detection method.\n");
		STATUS("I'm using 'zaef' for you.\n");
	}
	if ( strcmp(speaks, "zaef") == 0 ) {
		peaks = PEAK_ZAEF;
	} else if ( strcmp(speaks, "hdf5") == 0 ) {
		peaks = PEAK_HDF5;
	} else {
		ERROR("Unrecognised peak detection method '%s'\n", speaks);
		return 1;
	}
	free(speaks);

	if ( pdb == NULL ) {
		pdb = strdup("molecule.pdb");
	}

	if ( prefix == NULL ) {
		prefix = strdup("");
	} else {
		if ( config_checkprefix ) {
			prefix = check_prefix(prefix);
		}
	}

	if ( n_proc == 0 ) {
		ERROR("Invalid number of processes.\n");
		return 1;
	}

	if ( (indm_str == NULL) ||
	     ((indm_str != NULL) && (strcmp(indm_str, "none") == 0)) ) {
		STATUS("Not indexing anything.\n");
		indexer_needs_cell = 0;
		reduction_needs_cell = 0;
		indm = NULL;
		cellr = CELLR_NONE;
	} else {
		if ( indm_str == NULL ) {
			STATUS("You didn't specify an indexing method, so I "
			       " won't try to index anything.\n"
			       "If that isn't what you wanted, re-run with"
			       " --indexing=<method>.\n");
			indm = NULL;
			indexer_needs_cell = 0;
		} else {
			indm = build_indexer_list(indm_str,
			                          &indexer_needs_cell);
			if ( indm == NULL ) {
				ERROR("Invalid indexer list '%s'\n", indm_str);
				return 1;
			}
			free(indm_str);
		}

		reduction_needs_cell = 0;
		if ( scellr == NULL ) {
			STATUS("You didn't specify a cell reduction method, so"
			       " I'm going to use 'reduce'.\n");
			cellr = CELLR_REDUCE;
			reduction_needs_cell = 1;
		} else {
			int err;
			cellr = parse_cell_reduction(scellr, &err,
			                             &reduction_needs_cell);
			if ( err ) {
				ERROR("Unrecognised cell reduction '%s'\n",
			              scellr);
				return 1;
			}
			free(scellr);
		}
	}

	/* No indexing -> no reduction */
	if ( indm == NULL ) reduction_needs_cell = 0;

	if ( toler != NULL ) {
		int ttt;
		ttt = sscanf(toler, "%f,%f,%f,%f",
		             &tols[0], &tols[1], &tols[2], &tols[3] );
		if ( ttt != 4 ) {
			ERROR("Invalid parameters for '--tolerance'\n");
			return 1;
		}
	}

	if ( intrad != NULL ) {
		int r;
		r = sscanf(intrad, "%f,%f,%f", &ir_inn, &ir_mid, &ir_out);
		if ( r != 3 ) {
			ERROR("Invalid parameters for '--int-radius'\n");
			return 1;
		}
	} else {
		STATUS("WARNING: You did not specify --int-radius.\n");
		STATUS("WARNING: I will use the default values, which are"
		       " probably not appropriate for your patterns.\n");
	}

	if ( geometry == NULL ) {
		ERROR("You need to specify a geometry file with --geometry\n");
		return 1;
	}

	det = get_detector_geometry(geometry);
	if ( det == NULL ) {
		ERROR("Failed to read detector geometry from '%s'\n", geometry);
		return 1;
	}
	free(geometry);

	if ( reduction_needs_cell || indexer_needs_cell ) {
		cell = load_cell_from_pdb(pdb);
		if ( cell == NULL ) {
			ERROR("Couldn't read unit cell (from %s)\n", pdb);
			return 1;
		}
	} else {
		STATUS("No cell needed for these choices of indexing"
		       " and reduction.\n");
		cell = NULL;
	}
	free(pdb);

	write_stream_header(ofh, argc, argv);

	if ( beam != NULL ) {
		nominal_photon_energy = beam->photon_energy;
	} else {
		STATUS("No beam parameters file was given, so I'm taking the"
		       " nominal photon energy to be 2 keV.\n");
		ERROR("I'm also going to assume 1 ADU per photon, which is");
		ERROR(" almost certainly wrong.  Peak sigmas will be"
		      " incorrect.\n");
		nominal_photon_energy = 2000.0;
	}

	/* Get first filename and use it to set up the indexing */
	prepare_line = malloc(1024);
	rval = fgets(prepare_line, 1023, fh);
	if ( rval == NULL ) {
		ERROR("Failed to get filename to prepare indexing.\n");
		return 1;
	}
	use_this_one_instead = strdup(prepare_line);
	chomp(prepare_line);
	if ( config_basename ) {
		char *tmp;
		tmp = safe_basename(prepare_line);
		free(prepare_line);
		prepare_line = tmp;
	}
	snprintf(prepare_filename, 1023, "%s%s", prefix, prepare_line);
	free(prepare_line);

	/* Prepare the indexer */
	if ( indm != NULL ) {
		ipriv = prepare_indexing(indm, cell, prepare_filename, det,
		                         nominal_photon_energy);
		if ( ipriv == NULL ) {
			ERROR("Failed to prepare indexing.\n");
			return 1;
		}
	} else {
		ipriv = NULL;
	}

	gsl_set_error_handler_off();

	/* Static worker args */
	iargs.cell = cell;
	iargs.config_cmfilter = config_cmfilter;
	iargs.config_noisefilter = config_noisefilter;
	iargs.config_verbose = config_verbose;
	iargs.config_satcorr = config_satcorr;
	iargs.config_closer = config_closer;
	iargs.config_insane = config_insane;
	iargs.config_bgsub = config_bgsub;
	iargs.cellr = cellr;
	iargs.tols[0] = tols[0];
	iargs.tols[1] = tols[1];
	iargs.tols[2] = tols[2];
	iargs.tols[3] = tols[3];
	iargs.threshold = threshold;
	iargs.min_gradient = min_gradient;
	iargs.min_snr = min_snr;
	iargs.min_int_snr = min_int_snr;
	iargs.det = det;
	iargs.indm = indm;
	iargs.ipriv = ipriv;
	iargs.peaks = peaks;
	iargs.beam = beam;
	iargs.element = element;
	iargs.stream_flags = stream_flags;
	iargs.hdf5_peak_path = hdf5_peak_path;
	iargs.copyme = copyme;
	iargs.ir_inn = ir_inn;
	iargs.ir_mid = ir_mid;
	iargs.ir_out = ir_out;

	n_indexable = 0;
	n_processed = 0;
	n_indexable_last_stats = 0;
	n_processed_last_stats = 0;
	t_last_stats = get_monotonic_seconds();

	stream_pipe_read = calloc(n_proc, sizeof(int));
	stream_pipe_write = calloc(n_proc, sizeof(int));
	if ( stream_pipe_read == NULL ) {
		ERROR("Couldn't allocate memory for pipes.\n");
		return 1;
	}
	if ( stream_pipe_write == NULL ) {
		ERROR("Couldn't allocate memory for pipes.\n");
		return 1;
	}

	for ( i=0; i<n_proc; i++ ) {

		int stream_pipe[2];

		if ( pipe(stream_pipe) == - 1 ) {
			ERROR("pipe() failed!\n");
			return 1;
		}

		stream_pipe_read[i] = stream_pipe[0];
		stream_pipe_write[i] = stream_pipe[1];

	}

	pr = fork();
	if ( pr == - 1 ) {
		ERROR("fork() failed (for reader process)\n");
		return 1;
	}

	if ( pr == 0 ) {

		/* Free resources not needed by reader
		 * (but which will be needed by worker or master) */
		for ( i=0; i<n_proc; i++ ) {
			close(stream_pipe_write[i]);
		}
		free(prefix);
		free(use_this_one_instead);
		free(stream_pipe_write);
		cleanup_indexing(ipriv);
		free(indm);
		free(ipriv);
		free_detector_geometry(det);
		free(beam);
		free(element);
		free(hdf5_peak_path);
		free_copy_hdf5_field_list(copyme);
		cell_free(cell);
		fclose(fh);

		run_reader(stream_pipe_read, n_proc, ofh);

		free(stream_pipe_read);

		exit(0);

	}

	/* Set up signal handler to take action if any children die */
	sa.sa_flags = SA_SIGINFO | SA_NOCLDSTOP;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = signal_handler;
	r = sigaction(SIGCHLD, &sa, NULL);
	if ( r == -1 ) {
		ERROR("Failed to set signal handler!\n");
		return 1;
	}

	/* Free resources needed by reader only */
	if ( ofh != stdout ) fclose(ofh);
	for ( i=0; i<n_proc; i++ ) {
		close(stream_pipe_read[i]);
	}
	free(stream_pipe_read);

	filename_pipes = calloc(n_proc, sizeof(int));
	result_fhs = calloc(n_proc, sizeof(FILE *));
	pids = calloc(n_proc, sizeof(pid_t));
	if ( filename_pipes == NULL ) {
		ERROR("Couldn't allocate memory for pipes.\n");
		return 1;
	}
	if ( result_fhs == NULL ) {
		ERROR("Couldn't allocate memory for pipe file handles.\n");
		return 1;
	}
	if ( pids == NULL ) {
		ERROR("Couldn't allocate memory for PIDs.\n");
		return 1;
	}

	/* Fork the right number of times */
	for ( i=0; i<n_proc; i++ ) {

		pid_t p;
		int filename_pipe[2];
		int result_pipe[2];

		if ( pipe(filename_pipe) == - 1 ) {
			ERROR("pipe() failed!\n");
			return 1;
		}

		if ( pipe(result_pipe) == - 1 ) {
			ERROR("pipe() failed!\n");
			return 1;
		}

		p = fork();
		if ( p == -1 ) {
			ERROR("fork() failed!\n");
			return 1;
		}

		if ( p == 0 ) {

			FILE *sfh;
			int j;

			/* Free resources which will not be needed by worker */
			for ( j=0; j<n_proc; j++ ) {
				if ( i != j ) close(stream_pipe_write[j]);
			}
			free(prefix);
			free(use_this_one_instead);
			free(filename_pipes);
			free(result_fhs);
			fclose(fh);
			free(pids);
			for ( j=0; j<i-1; j++ ) {
				fclose(result_fhs[j]);
				close(filename_pipes[j]);
			}

			/* Child process gets the 'read' end of the filename
			 * pipe, and the 'write' end of the result pipe. */
			close(filename_pipe[1]);
			close(result_pipe[0]);

			sfh = fdopen(stream_pipe_write[i], "w");
			run_work(&iargs, filename_pipe[0], result_pipe[1],
			         sfh, i);
			fclose(sfh);

			free(stream_pipe_write);
			close(filename_pipe[0]);
			close(result_pipe[1]);

			exit(0);

		}

		/* Parent process gets the 'write' end of the filename pipe
		 * and the 'read' end of the result pipe. */
		pids[i] = p;
		close(filename_pipe[0]);
		close(result_pipe[1]);
		filename_pipes[i] = filename_pipe[1];

		result_fhs[i] = fdopen(result_pipe[0], "r");
		if ( result_fhs[i] == NULL ) {
			ERROR("fdopen() failed.\n");
			return 1;
		}

	}

	/* Free resources which will not be used by the main thread */
	cleanup_indexing(ipriv);
	free(indm);
	free(ipriv);
	free_detector_geometry(det);
	free(beam);
	free(element);
	free(hdf5_peak_path);
	free_copy_hdf5_field_list(copyme);
	cell_free(cell);
	for ( i=0; i<n_proc; i++ ) {
		close(stream_pipe_write[i]);
	}
	free(stream_pipe_write);

	/* Send first image to all children */
	for ( i=0; i<n_proc; i++ ) {

		char *nextImage;

		nextImage = get_pattern(fh, &use_this_one_instead,
		                        config_basename, prefix);

		if ( nextImage != NULL ) {

			write(filename_pipes[i], nextImage, strlen(nextImage));
			write(filename_pipes[i], "\n", 1);

			free(nextImage);

		} else {

			int r;

			/* No more files to process.. already? */
			r = write(filename_pipes[i], "\n", 1);
			if ( r < 0 ) {
				ERROR("Write pipe\n");
			}

		}

	}

	finished = calloc(n_proc, sizeof(int));
	if ( finished == NULL ) {
		ERROR("Couldn't allocate memory for process flags.\n");
		return 1;
	}

	allDone = 0;
	while ( !allDone ) {

		int r, i;
		struct timeval tv;
		fd_set fds;
		double tNow;
		int fdmax;

		tv.tv_sec = 5;
		tv.tv_usec = 0;

		FD_ZERO(&fds);
		fdmax = 0;
		for ( i=0; i<n_proc; i++ ) {

			int fd;

			if ( finished[i] ) continue;

			fd = fileno(result_fhs[i]);
			FD_SET(fd, &fds);
			if ( fd > fdmax ) fdmax = fd;

		}

		r = select(fdmax+1, &fds, NULL, NULL, &tv);

		if ( r == -1 ) {
			ERROR("select() failed: %s\n", strerror(errno));
			continue;
		}

		if ( r == 0 ) continue; /* No progress this time.  Try again */

		for ( i=0; i<n_proc; i++ ) {

			char *nextImage;
			char results[1024];
			char *rval;
			int fd;

			if ( finished[i] ) continue;

			fd = fileno(result_fhs[i]);
			if ( !FD_ISSET(fd, &fds) ) continue;

			rval = fgets(results, 1024, result_fhs[i]);
			if ( rval == NULL ) {
				if ( feof(result_fhs[i]) ) {
					/* Process died */
					finished[i] = 1;
				} else {
					ERROR("fgets() failed: %s\n",
					      strerror(errno));
				}
				continue;
			}

			chomp(results);
			n_indexable += atoi(results);
			n_processed++;

			/* Send next filename */
			nextImage = get_pattern(fh, &use_this_one_instead,
	                                        config_basename, prefix);

			if ( nextImage == NULL ) {
				/* No more images */
				r = write(filename_pipes[i], "\n", 1);
				if ( r < 0 ) {
					ERROR("Write pipe\n");
				}
			} else {

				r = write(filename_pipes[i], nextImage,
				          strlen(nextImage));
				r -= write(filename_pipes[i], "\n", 1);
				if ( r < 0 ) {
					ERROR("write pipe\n");
				}
				free(nextImage);
			}

		}

		/* Update progress */
		tNow = get_monotonic_seconds();
		if ( tNow >= t_last_stats+STATS_EVERY_N_SECONDS ) {

			STATUS("%i out of %i indexed so far,"
			       " %i out of %i since the last message.\n",
			       n_indexable, n_processed,
			       n_indexable - n_indexable_last_stats,
			       n_processed - n_processed_last_stats);

			n_indexable_last_stats = n_indexable;
			n_processed_last_stats = n_processed;
			t_last_stats = tNow;

		}

		allDone = 1;
		for ( i=0; i<n_proc; i++ ) {
			if ( !finished[i] ) allDone = 0;
		}

	}

	fclose(fh);

	for ( i=0; i<n_proc; i++ ) {
		int status;
		waitpid(pids[i], &status, 0);
	}

	for ( i=0; i<n_proc; i++ ) {
		close(filename_pipes[i]);
		fclose(result_fhs[i]);
	}

	free(prefix);
	free(filename_pipes);
	free(result_fhs);
	free(pids);
	free(finished);

	STATUS("There were %i images, of which %i could be indexed.\n",
	       n_processed, n_indexable);

	return 0;
}
