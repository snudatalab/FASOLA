import os
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import fasola
from models import get_dcase_model


def load_data(path):
    """
    Load data from a pickle file and normalize it.

    Args:
        path (str): Path to the pickle file.

    Returns:
        tuple: (Normalized Features X, Labels y)
    """
    print(f"Loading Data: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        X, y = data["data"], data["label"]
    else:
        X, y = data

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-6)
    return X, y


# ==============================================================================
# Main
# ==============================================================================
def main():
    """
    Main execution function for FASOLA adaptation.
    """
    parser = argparse.ArgumentParser()

    # data / env
    parser.add_argument("--targets", nargs="+",
                        default=["s1", "s2", "s3", "s4", "s5", "s6"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="Encoded_data_dcase")
    parser.add_argument("--model_dir", type=str,
                        default="saved_models/dcase2020/individual_sources/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # FASOLA
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--sharpness", type=float, default=10.0)
    parser.add_argument("--em_iters", type=int, default=5)
    parser.add_argument("--momentum_alpha", type=int, default=0.2)

    parser.add_argument("--lr", type=float, default=1e-2)

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------
    fasola.set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    results_db = {}

    for target in args.targets:
        print(f"Target Domain: {target}")

        # -------------------------------
        # Load target data
        # -------------------------------
        data_path = os.path.join(args.data_dir, f"{target}.pkl")
        if not os.path.exists(data_path):
            print(f"Missing data: {target}")
            continue

        X, y = load_data(data_path)
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
        )

        # -------------------------------
        # Load source models
        # -------------------------------
        source_models = []
        model_paths = glob.glob(os.path.join(args.model_dir, "cp-resnet_*.pth"))

        for path in model_paths:
            name = os.path.basename(path).replace(".pth", "").replace("cp-resnet_", "")
            if name == target:
                continue

            model = get_dcase_model("cp-resnet", 10, 128).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            source_models.append(model)

        if len(source_models) == 0:
            print("No source models loaded.")
            continue
        acc, f1 = fasola.run_fasola(
            source_models,
            loader,
            device,
            tau=args.tau,
            sharpness=args.sharpness,
            em_iters = args.em_iters,
            momentum_alpha=args.momentum_alpha
        )
        results_db.setdefault("FASOLA", {})[target] = {
            "acc": acc,
            "f1": f1,
        }
    # --------------------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------------------
    rows = []
    row = {"Method": "FASOLA"}
    for target in args.targets:
        if target in results_db.get('FASOLA', {}):
            res = results_db['FASOLA'][target]
            row[f"{target}_Acc"] = round(res["acc"], 2)
            row[f"{target}_F1"] = round(res["f1"], 2)
        else:
            row[f"{target}_Acc"] = "-"
            row[f"{target}_F1"] = "-"
    rows.append(row)

    df_flat = pd.DataFrame(rows)

    final_cols = pd.MultiIndex.from_product(
        [args.targets + ["Average"], ["Acc", "F1"]],
        names=["Dataset", "Metric"],
    )
    df_final = pd.DataFrame(index=df_flat["Method"], columns=final_cols)

    for _, row in df_flat.iterrows():
        m = row["Method"]
        for t in args.targets:
            df_final.loc[m, (t, "Acc")] = row[f"{t}_Acc"]
            df_final.loc[m, (t, "F1")] = row[f"{t}_F1"]

    df_numeric = df_final.apply(pd.to_numeric, errors="coerce")
    df_final[("Average", "Acc")] = df_numeric.loc[:, (slice(None), "Acc")].mean(axis=1).round(2)
    df_final[("Average", "F1")] = df_numeric.loc[:, (slice(None), "F1")].mean(axis=1).round(2)

    print("=" * 70)
    print(df_final)


if __name__ == "__main__":
    main()
