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
    std::vector<float> params;
    torch::jit::script::Module module;
    torch::DeviceType device_type = torch::kCPU;
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

    UnitCell *uc = NULL;
    {
        // thread-local guard that disabled gradient calculation.
        torch::NoGradGuard no_grad;

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(
            torch::from_blob(peaks, {1, npk, 3}, torch::dtype(torch::kFloat64))
            .to(torch::kFloat32)
            .to(prv_data->device_type) * 1e-10);
        inputs.push_back(torch::from_blob(cell, {3, 3}).to(prv_data->device_type));
        inputs.push_back(torch::from_blob(prv_data->params.data(), {prv_data->params.size()}).to(prv_data->device_type));

        // Execute the model and turn its output into a tuple of tensors.
        prv_data->module.to(prv_data->device_type);
        auto output = prv_data->module(inputs).toTuple()->elements();

        if (output[0].toTensor().size(0) == 0 ||
            at::isnan(output[1].toTensor()).any().item<bool>()) {
            // std::cout << "input0:\n" << inputs[0] << '\n';
            // std::cout << "input1:\n" << inputs[1] << '\n';
            // STATUS("torchidx: crystal not found for %s\n", image->ev);
            return 0;
        }

        // std::cout << "output0:\n" << output[0] << '\n';
        // std::cout << "output1:\n" << output[1] << '\n';

        // transpose and scale
        auto cell_m = output[1].toTensor()[0][0].transpose(0, 1) * 1e-10;

        auto ocell = cell_m.accessor<float,2>();
        uc = cell_new();
        cell_set_cartesian(uc,
                            ocell[0][0], ocell[1][0], ocell[2][0],
                            ocell[0][1], ocell[1][1], ocell[2][1],
                            ocell[0][2], ocell[1][2], ocell[2][2]);

        makeRightHanded(uc);
        cell_set_lattice_type(uc, cell_get_lattice_type(prv_data->cellTemplate));
        cell_set_centering(uc, cell_get_centering(prv_data->cellTemplate));
        cell_set_unique_axis(uc, cell_get_unique_axis(prv_data->cellTemplate));
    }

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


static std::vector<float> parse_params(char *cvs_params)
{
    std::vector<float> params;
    std::stringstream ss(cvs_params);
    float i;

    while (ss >> i) {
        params.push_back(i);
        if (ss.peek() == ',') ss.ignore();
    }
    return params;
}

void *torchidx_prepare(IndexingMethod *indm, UnitCell *cell, struct torchidx_options *opts) {
    if ( cell == NULL ) {
        ERROR("Unit cell information is required for fast feedback indexer.\n");
        return NULL;
    }

    struct torchidx_private_data *prv_data = new torchidx_private_data;
    prv_data->cellTemplate = cell;
    prv_data->opts = *opts;
    torch::set_num_threads(prv_data->opts.num_threads);

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        STATUS("Loading model %s.\n", opts->filename);
        prv_data->module = torch::jit::load(opts->filename);
    } catch (const c10::Error& e) {
        ERROR("Error loading the model %s.\n", opts->filename);
        return NULL;
    }

    if (opts->params != NULL) {
        STATUS("Parsing model parameters '%s'.\n", opts->params);
        prv_data->params = parse_params(opts->params);
    }

    // if (torch::cuda::is_available()) {
    //     std::cout << "CUDA available! Running on GPU." << std::endl;
    //     prv_data->device_type = torch::kCUDA;
    // } else {
        std::cout << "Running on CPU." << std::endl;
        prv_data->device_type = torch::kCPU;
    // }

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
           "     --torchidx-filename\n"
           "                            TorchScript filename.\n"
           "     --torchidx-params\n"
           "                            Comma separated list of parameters as required by the TorchScript module.\n"
           "     --torchidx-num-threads\n"
           "                            Number of threads on each worker process.\n"
           "                            Default: 1\n"
    );
}


int torchidx_default_options(struct torchidx_options **opts_ptr)
{
    struct torchidx_options *opts = new torchidx_options();
    if ( opts == NULL ) return ENOMEM;

    opts->filename = NULL;
    opts->params = NULL;
    opts->num_threads = 1;
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
            STATUS("--torchidx-filename %s \n", arg);
            (*opts_ptr)->filename = arg;
            break;

        case 3 :
            STATUS("--torchidx-params %s \n", arg);
            (*opts_ptr)->params = arg;
            break;
    }

    return 0;
}


static struct argp_option torchidx_options[] = {
        {"help-torchidx", 1, NULL, OPTION_NO_USAGE, "Show options for fast feedback indexing algorithm", 99},
        {"torchidx-filename", 2, "torchidx_filename", OPTION_HIDDEN, NULL},
        {"torchidx-params", 3, "torchidx_params", OPTION_HIDDEN, NULL},
        {"torchidx-num-threads", 4, "torchidx_num_threads", OPTION_HIDDEN, NULL},
        {0}
};


struct argp torchidx_argp = { torchidx_options, torchidx_parse_arg, NULL, NULL, NULL, NULL, NULL };
