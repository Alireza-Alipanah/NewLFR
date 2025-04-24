import wandb

from utils import (
    load_wb_data,
    load_convex_combination_data,
    load_dsprite_data,
    load_dominos_cmnist_all_exps_data,
)

import os


EXPERIMENT_CONFIGS = {

    "NormalWaterbirds": {
        "loader": load_wb_data,
        "loader_kwargs": lambda args: {
            "basedir": args.base_dir,
            "batch_size": args.batch_size,
            "spurious_correlation_value": args.spurious_correlation_value,
            "spurious_correlation_value_validation": args.spurious_correlation_value_validation,
        },
        "path_parts": lambda args: [wandb.run.name or wandb.run.id],
    },

    "CifarMnistConvexCombination": {
        "loader": load_convex_combination_data,
        "loader_kwargs": lambda args: {
            "mapping_csv_path": os.path.join(
                args.cc_mapping_basedir,
                f"{args.cc_correlation}_mapping.csv"
            ),
            "alpha": args.cc_alpha,
            "correlation": args.cc_correlation,
            "batch_size": args.batch_size,
        },
        "path_parts": lambda args: [
            f"alpha_{args.cc_alpha}",
            f"seed_{args.seed}",
            f"correlation_{int(float(args.cc_correlation) * 100)}",
        ],
    },

    "DSprite": {
        "loader": load_dsprite_data,
        "loader_kwargs": lambda args: {
            "args": args,
            "basedir": args.base_dir,
            "batch_size": args.batch_size,
            "spurious_correlation_value": args.spurious_correlation_value / 100,
            "spurious_correlation_value_validation": args.spurious_correlation_value_validation / 100,
        },
        "path_parts": lambda args: [
            "DSprite",
            f"seed_{args.seed}",
            f"sc_{args.spurious_correlation_value}",
            f"batch_size_{args.batch_size}",
        ],
    },

    "Dominos_Cmnist_all_exps": {
        "loader": load_dominos_cmnist_all_exps_data,
        "loader_kwargs": lambda args: {
            "args": args,
            "batch_size": args.batch_size,
        },
        "path_parts": lambda args: [
            "Dominos_Cmnist_all_exps",
            f"seed_{args.seed}",
            f"dataset_total_count_{args.dataset_total_count}",
            f"color_correlation_{args.color_correlation}",
            f"shape_correlation_{args.shape_correlation}",
            f"batch_size_{args.batch_size}",
        ],
    },
}