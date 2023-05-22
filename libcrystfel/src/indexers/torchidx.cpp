/*
 * torchidx.c
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

#include "torchidx.h"
#include "cell-utils.h"

#include <libcrystfel-config.h>
#include <stdio.h>
#include <stdlib.h>


#ifdef HAVE_TORCH

struct torchidx_private_data {
    UnitCell *cellTemplate;
    struct torchidx_options opts;
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

int run_torchidx(struct image *image, void *ipriv) {
    int npk;
    int i;

    struct torchidx_private_data *prv_data = (struct torchidx_private_data *) ipriv;

    npk = image_feature_count(image->features);
    if ( npk < prv_data->opts.min_peaks )
        return 0;

    double peaks[npk][3];
    for ( i=0; i<npk; i++ ) {
        struct imagefeature *f = image_get_feature(image->features, i);
        if ( f == NULL ) {
            ERROR("Empty feature ???");
            continue;
        }

        detgeom_transform_coords(&image->detgeom->panels[f->pn],
                                 f->fs, f->ss, image->lambda,
                                 0.0, 0.0, peaks[i]);
    }

    float cell[9];
    double cell_internal_double[9];

    cell_get_cartesian(prv_data->cellTemplate,
                       &cell_internal_double[0],&cell_internal_double[1],&cell_internal_double[2],
                       &cell_internal_double[3],&cell_internal_double[4],&cell_internal_double[5],
                       &cell_internal_double[6],&cell_internal_double[7],&cell_internal_double[8]);

    for (int i = 0; i < 9; i++)
        cell[i] = cell_internal_double[i] * 1e10;

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::from_blob(peaks, {1, npk, 3}, torch::dtype(torch::kFloat64)).to(torch::kFloat32) * 1e-10);
    inputs.push_back(torch::from_blob(cell, {3, 3}));
    // inputs.push_back(torch::from_blob(peaks, {1, npk, 3}, torch::dtype(torch::kFloat64)) * 1e-10);
    // inputs.push_back(torch::from_blob(cell_internal_double, {3, 3}, torch::dtype(torch::kFloat64)) * 1e10);
    inputs.push_back(prv_data->opts.min_peaks);
    inputs.push_back(180);
    inputs.push_back(prv_data->opts.num_candidate_vectors);

    // Execute the model and turn its output into a tuple of tensors.
    auto output = prv_data->module(inputs).toTuple()->elements();
    if (output[0].toTensor().size(0) == 0) {
        // std::cout << "input0:\n" << inputs[0] << '\n';
        // std::cout << "input1:\n" << inputs[1] << '\n';
        STATUS("torchidx: crystal not found for %s\n", image->ev);
        return 0;
    }

    // std::cout << "output0:\n" << output[0] << '\n';
    // std::cout << "output1:\n" << output[1] << '\n';

    // transpose and scale
    auto cell_m = output[1].toTensor()[0][0].transpose(0, 1) * 1e-10;

    auto ocell = cell_m.accessor<float,2>();
    UnitCell *uc;
    uc = cell_new();
    cell_set_cartesian(uc,
                        ocell[0][0], ocell[1][0], ocell[2][0],
                        ocell[0][1], ocell[1][1], ocell[2][1],
                        ocell[0][2], ocell[1][2], ocell[2][2]);
    makeRightHanded(uc);

    cell_set_lattice_type(uc, cell_get_lattice_type(prv_data->cellTemplate));
    cell_set_centering(uc, cell_get_centering(prv_data->cellTemplate));
    cell_set_unique_axis(uc, cell_get_unique_axis(prv_data->cellTemplate));

    // double cell_internal_double[9];
    // cell_get_cartesian(uc,
    //                    &cell_internal_double[0],&cell_internal_double[1],&cell_internal_double[2],
    //                    &cell_internal_double[3],&cell_internal_double[4],&cell_internal_double[5],
    //                    &cell_internal_double[6],&cell_internal_double[7],&cell_internal_double[8]);
    // ERROR("torch cell: [%e %e %e], [%e %e %e], [%e %e %e] %s\n",
    //                    cell_internal_double[0],cell_internal_double[1],cell_internal_double[2],
    //                    cell_internal_double[3],cell_internal_double[4],cell_internal_double[5],
    //                    cell_internal_double[6],cell_internal_double[7],cell_internal_double[8], image->ev);

    if ( validate_cell(uc) ) {
        ERROR("torchidx: problem with returned cell!\n");
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

void *torchidx_prepare(IndexingMethod *indm, UnitCell *cell, struct torchidx_options *opts) {
    if ( cell == NULL ) {
        ERROR("Unit cell information is required for fast feedback indexer.\n");
        return NULL;
    }

    // struct torchidx_private_data *prv_data = (struct torchidx_private_data *) malloc(sizeof(struct torchidx_private_data));
    struct torchidx_private_data *prv_data = new torchidx_private_data;

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

void torchidx_cleanup(void *ipriv) {
    struct torchidx_private_data *prv_data = (struct torchidx_private_data *) ipriv;
    delete(prv_data);
}

const char *torchidx_probe(UnitCell *cell) {
    return "torchidx";
}

#else

int run_torchidx(struct image *image, void *ipriv)
{
	ERROR("This copy of CrystFEL was compiled without TORCHIDX support.\n");
	return 0;
}


void *torchidx_prepare(IndexingMethod *indm, UnitCell *cell, struct torchidx_options *opts)
{
	ERROR("This copy of CrystFEL was compiled without TORCHIDX support.\n");
	ERROR("To use TORCHIDX indexing, recompile with TORCHIDX.\n");
	return NULL;
}


void torchidx_cleanup(void *pp)
{
}


const char *torchidx_probe(UnitCell *cell)
{
	return NULL;
}

#endif

static void torchidx_show_help()
{
    printf("Parameters for the fast feedback indexing algorithm:\n"
           "     --torchidx-max-peaks\n"
           "                            Maximum number of peaks used for indexing.\n"
           "                            All peaks are used for refinement.\n"
           "                            Default: 250\n"
           "     --torchidx-min-peaks\n"
           "                            Maximum number of indexed peaks to accept solution.\n"
           "                            Default: 9\n"
           "     --torchidx-threshold\n"
           "                            Threshold to accept solution as indexed.\n"
           "                            Default: 0.02\n"
           "     --torchidx-output-cells\n"
           "                            Number of output cells.\n"
           "                            Default: 1\n"
           "     --torchidx-sample-points\n"
           "                            Number of sample points.\n"
           "                            Default: 32768\n"
           "     --torchidx-filename\n"
           "                            TorchScript filename.\n"

    );
}


int torchidx_default_options(struct torchidx_options **opts_ptr)
{
    struct torchidx_options *opts = new torchidx_options();
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


static error_t torchidx_parse_arg(int key, char *arg, struct argp_state *state)
{
    struct torchidx_options **opts_ptr = static_cast<struct torchidx_options**>(state->input);
    int r;

    switch ( key ) {
        case ARGP_KEY_INIT :
            r = torchidx_default_options(opts_ptr);
            if ( r ) return r;
            break;

        case 1 :
            torchidx_show_help();
            return EINVAL;

        case 2 :
            if (sscanf(arg, "%u", &(*opts_ptr)->max_peaks) != 1) {
                ERROR("Invalid value for --torchidx-max-peaks\n");
                return EINVAL;
            }
            break;

        case 3 :
            if (sscanf(arg, "%u", &(*opts_ptr)->min_peaks) != 1) {
                ERROR("Invalid value for --torchidx-min-peaks\n");
                return EINVAL;
            }
            break;

        case 4 :
            if (sscanf(arg, "%f", &(*opts_ptr)->threshold_for_solution) != 1) {
                ERROR("Invalid value for --torchidx-threshold\n");
                return EINVAL;
            }
            if (((*opts_ptr)->threshold_for_solution <= 0.0f) || ((*opts_ptr)->threshold_for_solution > 1.0f)) {
                ERROR("Invalid value for --torchidx-threshold; must be in range 0.0-1.0\n");
                return EINVAL;
            }
            break;
        case 5 :
            if (sscanf(arg, "%u", &(*opts_ptr)->output_cells) != 1) {
                ERROR("Invalid value for --torchidx-output-cells\n");
                return EINVAL;
            }
            if (((*opts_ptr)->output_cells <= 0) || ((*opts_ptr)->output_cells > 128)) {
                ERROR("Invalid value for --torchidx-output-cells; must be in range 1-128\n");
                return EINVAL;
            }
            break;
        case 6 :
            if (sscanf(arg, "%u", &(*opts_ptr)->sample_points) != 1) {
                ERROR("Invalid value for --torchidx-sample-points\n");
                return EINVAL;
            }
            break;
        case 7 :
            STATUS("--torchidx-filename %s \n", arg);
            // if (sscanf(arg, "%s", (*opts_ptr)->filename) != 1) {
            //     ERROR("Invalid value for --torchidx-filename\n");
            //     return EINVAL;
            // }
            (*opts_ptr)->filename = arg;
            STATUS("--torchidx-filename %s \n", (*opts_ptr)->filename);
            break;
    }

    return 0;
}


static struct argp_option torchidx_options[] = {
        {"help-torchidx", 1, NULL, OPTION_NO_USAGE, "Show options for fast feedback indexing algorithm", 99},
        {"torchidx-max-peaks", 2, "torchidx_maxn", OPTION_HIDDEN, NULL},
        {"torchidx-min-peaks", 3, "torchidx_minn", OPTION_HIDDEN, NULL},
        {"torchidx-threshold", 4, "torchidx_threshold", OPTION_HIDDEN, NULL},
        {"torchidx-output-cells", 5, "torchidx_out_cells", OPTION_HIDDEN, NULL},
        {"torchidx-sample-points", 6, "torchidx_sample_points", OPTION_HIDDEN, NULL},
        {"torchidx-filename", 7, "torchidx_filename", OPTION_HIDDEN, NULL},
        {0}
};


struct argp torchidx_argp = { torchidx_options, torchidx_parse_arg, NULL, NULL, NULL, NULL, NULL };
