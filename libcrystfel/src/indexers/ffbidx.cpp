/*
 * ffbidx.c
 *
 * Invoke the Fast Feedback Indexer library
 *
 * Copyright © 2023 Paul Scherrer Institute
 * Copyright © 2017-2021 Deutsches Elektronen-Synchrotron DESY,
 *                       a research centre of the Helmholtz Association.
 *
 * Authors:
 *   2023 Filip Leonarski <filip.leonarski@psi.ch>
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
#include <torch/torch.h>
#include <torch/script.h>

#include "ffbidx.h"
#include "cell-utils.h"

#include <libcrystfel-config.h>
#include <stdio.h>
#include <stdlib.h>


// #ifdef HAVE_TORCH

struct ffbidx_private_data {
    UnitCell *cellTemplate;
    struct ffbidx_options opts;
    torch::jit::script::Module module;
};

static void makeRightHanded(UnitCell *cell)
{
    // From xgandalf.c
    double ax, ay, az, bx, by, bz, cx, cy, cz;
    cell_get_cartesian(cell, &ax, &ay, &az, &bx, &by, &bz, &cx, &cy, &cz);

    if ( !right_handed(cell) ) {
        cell_set_cartesian(cell, -ax, -ay, -az, -bx, -by, -bz, -cx, -cy, -cz);
    }
}

int run_ffbidx(struct image *image, void *ipriv) {
    int npk;
    int i;

    struct ffbidx_private_data *prv_data = (struct ffbidx_private_data *) ipriv;

    npk = image_feature_count(image->features);
    if ( npk < prv_data->opts.min_peaks )
        return 0;

    float *x = (float *) calloc(npk, sizeof(float));
    float *y = (float *) calloc(npk, sizeof(float));
    float *z = (float *) calloc(npk, sizeof(float));

    for ( i=0; i<npk; i++ ) {

        struct imagefeature *f;
        double r[3];

        f = image_get_feature(image->features, i);
        if ( f == NULL ) {
            ERROR("Empty feature ???");
            continue;
        }

        detgeom_transform_coords(&image->detgeom->panels[f->pn],
                                 f->fs, f->ss, image->lambda,
                                 0.0, 0.0, r);
        x[i] = r[0] * 1e-10;
        y[i] = r[1] * 1e-10;
        z[i] = r[2] * 1e-10;
    }


    float cell[9];

    double cell_internal_double[9];

    cell_get_cartesian(prv_data->cellTemplate,
                       &cell_internal_double[0],&cell_internal_double[1],&cell_internal_double[2],
                       &cell_internal_double[3],&cell_internal_double[4],&cell_internal_double[5],
                       &cell_internal_double[6],&cell_internal_double[7],&cell_internal_double[8]);

    for (int i = 0; i < 9; i++)
        cell[i] = cell_internal_double[i] * 1e10;

    // struct ffbidx_settings settings;
    // settings.cpers_max_spots = prv_data->opts.max_peaks;
    // settings.cifss_min_spots = prv_data->opts.min_peaks;
    // settings.cvc_threshold = prv_data->opts.threshold_for_solution;
    // settings.cpers_max_output_cells = prv_data->opts.output_cells;
    // settings.crt_num_sample_points = prv_data->opts.sample_points;
    // settings.cpers_num_candidate_vectors = prv_data->opts.num_candidate_vectors;
    
    // fast_feedback_crystfel(&settings, cell, x, y, z, npk);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::from_blob(cell, {3, 3}));

    std::cout << "inputs:\n" << inputs[0] << '\n';

    // Execute the model and turn its output into a tensor.
    at::Tensor output = prv_data->module(inputs).toTensor();
    std::cout << "output:\n" << output << '\n';
    // STATUS()

    free(x);
    free(y);
    free(z);

    UnitCell *uc;

    uc = cell_new();

    cell_set_cartesian(uc,
                        cell[0] * 1e-10, cell[3] * 1e-10, cell[6] * 1e-10,
                        cell[1] * 1e-10, cell[4] * 1e-10, cell[7] * 1e-10,
                        cell[2] * 1e-10, cell[5] * 1e-10, cell[8] * 1e-10);

    makeRightHanded(uc);

    cell_set_lattice_type(uc, cell_get_lattice_type(prv_data->cellTemplate));
    cell_set_centering(uc, cell_get_centering(prv_data->cellTemplate));
    cell_set_unique_axis(uc, cell_get_unique_axis(prv_data->cellTemplate));

    if ( validate_cell(uc) ) {
        ERROR("ffbidx: problem with returned cell!\n");
        cell_free(uc);
        return 0;
    }

    Crystal *cr = crystal_new();
    if ( cr == NULL ) {
        ERROR("Failed to allocate crystal.\n");
        return 0;
    }
    crystal_set_cell(cr, uc);
    crystal_set_det_shift(cr, 0, 0);
    image_add_crystal(image, cr);
    return 1;
}

void *ffbidx_prepare(IndexingMethod *indm, UnitCell *cell, struct ffbidx_options *opts) {
    if ( cell == NULL ) {
        ERROR("Unit cell information is required for fast feedback indexer.\n");
        return NULL;
    }

    // struct ffbidx_private_data *prv_data = (struct ffbidx_private_data *) malloc(sizeof(struct ffbidx_private_data));
    struct ffbidx_private_data *prv_data = new ffbidx_private_data;

    prv_data->cellTemplate = cell;
    prv_data->opts = *opts;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        STATUS("Loading model %s.\n", opts->filename);
        prv_data->module = torch::jit::load(opts->filename);
        // torch::jit::script::Module module = torch::jit::load(opts->filename);
        // prv_data->module = module;
        STATUS("Loaded model %s.\n", opts->filename);
    }
    catch (const c10::Error& e) {
        ERROR("Error loading the model %s.\n", opts->filename);
        return NULL;
    }

    // *indm &= INDEXING_METHOD_MASK | INDEXING_USE_CELL_PARAMETERS;
    return prv_data;
}

void ffbidx_cleanup(void *ipriv) {
    struct ffbidx_private_data *prv_data = (struct ffbidx_private_data *) ipriv;
    delete(prv_data);
}

const char *ffbidx_probe(UnitCell *cell) {
    return "ffbidx";
}

// #else

// int run_ffbidx(struct image *image, void *ipriv)
// {
// 	ERROR("This copy of CrystFEL was compiled without FFBIDX support.\n");
// 	return 0;
// }


// void *ffbidx_prepare(IndexingMethod *indm, UnitCell *cell, struct ffbidx_options *opts)
// {
// 	ERROR("This copy of CrystFEL was compiled without FFBIDX support.\n");
// 	ERROR("To use FFBIDX indexing, recompile with FFBIDX.\n");
// 	return NULL;
// }


// void ffbidx_cleanup(void *pp)
// {
// }


// const char *ffbidx_probe(UnitCell *cell)
// {
// 	return NULL;
// }

// #endif

static void ffbidx_show_help()
{
    printf("Parameters for the fast feedback indexing algorithm:\n"
           "     --ffbidx-max-peaks\n"
           "                            Maximum number of peaks used for indexing.\n"
           "                            All peaks are used for refinement.\n"
           "                            Default: 250\n"
           "     --ffbidx-min-peaks\n"
           "                            Maximum number of indexed peaks to accept solution.\n"
           "                            Default: 9\n"
           "     --ffbidx-threshold\n"
           "                            Threshold to accept solution as indexed.\n"
           "                            Default: 0.02\n"
           "     --ffbidx-output-cells\n"
           "                            Number of output cells.\n"
           "                            Default: 1\n"
           "     --ffbidx-sample-points\n"
           "                            Number of sample points.\n"
           "                            Default: 32768\n"
           "     --ffbidx-filename\n"
           "                            TorchScript filename.\n"

    );
}


int ffbidx_default_options(struct ffbidx_options **opts_ptr)
{
    struct ffbidx_options *opts = new ffbidx_options();
    if ( opts == NULL ) return ENOMEM;

    opts->max_peaks = 100;
    opts->min_peaks = 9;
    opts->threshold_for_solution = 0.02f;
    opts->output_cells = 1;
    opts->sample_points = 32*1024;
    opts->num_candidate_vectors = 32;
    opts->filename = NULL;
    *opts_ptr = opts;
    return 0;
}


static error_t ffbidx_parse_arg(int key, char *arg, struct argp_state *state)
{
    struct ffbidx_options **opts_ptr = static_cast<struct ffbidx_options**>(state->input);
    int r;

    switch ( key ) {
        case ARGP_KEY_INIT :
            r = ffbidx_default_options(opts_ptr);
            if ( r ) return r;
            break;

        case 1 :
            ffbidx_show_help();
            return EINVAL;

        case 2 :
            if (sscanf(arg, "%u", &(*opts_ptr)->max_peaks) != 1) {
                ERROR("Invalid value for --ffbidx-max-peaks\n");
                return EINVAL;
            }
            break;

        case 3 :
            if (sscanf(arg, "%u", &(*opts_ptr)->min_peaks) != 1) {
                ERROR("Invalid value for --ffbidx-min-peaks\n");
                return EINVAL;
            }
            break;

        case 4 :
            if (sscanf(arg, "%f", &(*opts_ptr)->threshold_for_solution) != 1) {
                ERROR("Invalid value for --ffbidx-threshold\n");
                return EINVAL;
            }
            if (((*opts_ptr)->threshold_for_solution <= 0.0f) || ((*opts_ptr)->threshold_for_solution > 1.0f)) {
                ERROR("Invalid value for --ffbidx-threshold; must be in range 0.0-1.0\n");
                return EINVAL;
            }
            break;
        case 5 :
            if (sscanf(arg, "%u", &(*opts_ptr)->output_cells) != 1) {
                ERROR("Invalid value for --ffbidx-output-cells\n");
                return EINVAL;
            }
            if (((*opts_ptr)->output_cells <= 0) || ((*opts_ptr)->output_cells > 128)) {
                ERROR("Invalid value for --ffbidx-output-cells; must be in range 1-128\n");
                return EINVAL;
            }
            break;
        case 6 :
            if (sscanf(arg, "%u", &(*opts_ptr)->sample_points) != 1) {
                ERROR("Invalid value for --ffbidx-sample-points\n");
                return EINVAL;
            }
            break;
        case 7 :
            STATUS("--ffbidx-filename %s \n", arg);
            // if (sscanf(arg, "%s", (*opts_ptr)->filename) != 1) {
            //     ERROR("Invalid value for --ffbidx-filename\n");
            //     return EINVAL;
            // }
            (*opts_ptr)->filename = arg;
            STATUS("--ffbidx-filename %s \n", (*opts_ptr)->filename);
            break;
    }

    return 0;
}


static struct argp_option ffbidx_options[] = {
        {"help-ffbidx", 1, NULL, OPTION_NO_USAGE, "Show options for fast feedback indexing algorithm", 99},
        {"ffbidx-max-peaks", 2, "ffbidx_maxn", OPTION_HIDDEN, NULL},
        {"ffbidx-min-peaks", 3, "ffbidx_minn", OPTION_HIDDEN, NULL},
        {"ffbidx-threshold", 4, "ffbidx_threshold", OPTION_HIDDEN, NULL},
        {"ffbidx-output-cells", 5, "ffbidx_out_cells", OPTION_HIDDEN, NULL},
        {"ffbidx-sample-points", 6, "ffbidx_sample_points", OPTION_HIDDEN, NULL},
        {"ffbidx-filename", 7, "ffbidx_filename", OPTION_HIDDEN, NULL},
        {0}
};


struct argp ffbidx_argp = { ffbidx_options, ffbidx_parse_arg, NULL, NULL, NULL, NULL, NULL };
