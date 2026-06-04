import os
import pickle
import numpy as np
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from models import get_dcase_model

def get_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dcase", help="dcase, sti, zenodo")
    parser.add_argument("--model_type", type=str, default="cp-resnet", help="cp-resnet, tc-resnet")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    return parser.parse_args()

class AudioDataset(Dataset):
    """
    PyTorch Dataset for Audio Features.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return data, label

def load_pkl(path):
    """
    Load data from a pickle file.
    """
    print(f"   Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict): return data['data'], data['label']
    return data[0], data[1]

def get_normalization_stats(X):
    """
    Compute Mean and Std of the training data.
    """
    return np.mean(X), np.std(X)

def normalize(X, mean, std):
    """
    Normalize data using precomputed mean and std.
    """
    return (X - mean) / (std + 1e-6)

def main():
    """
    Main function to train individual source models.
    """
    args = get_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.data == "dcase":
        input_dir = "Encoded_data_dcase"
        dataset_name = "dcase2020"
        sources = ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
        
    else:
        raise ValueError(f"Unknown dataset: {args.data}")

    save_root = os.path.join("saved_models", dataset_name, "individual_sources")
    os.makedirs(save_root, exist_ok=True)

    print(f"Experiment: {args.model_type} on {dataset_name}")
    print(f"Load Dir: {input_dir}")
    print(f"Save Dir: {save_root}")

    for src in sources:
        print(f"\n{'='*40}")
        print(f"Training Source: [{src}]")
        print(f"{'='*40}")

        file_path = os.path.join(input_dir, f"{src}.pkl")
        if not os.path.exists(file_path):
            print(f"File missing: {file_path}, skipping...")
            continue

        X, y = load_pkl(file_path)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        # 2. Train/Valid Split (9:1)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(len(X) * 0.9)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mean, std = get_normalization_stats(X_train)
        X_train = normalize(X_train, mean, std)
        X_val = normalize(X_val, mean, std)
        print(f"   Norm Stats -> Mean: {mean:.2f}, Std: {std:.2f}")

        train_ds = AudioDataset(X_train, y_train)
        val_ds = AudioDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        n_classes = len(np.unique(y))
        input_dim = X_train.shape[1]

        model = get_dcase_model(
            model_name=args.model_type,
            num_classes=n_classes,
            input_dim=input_dim
        ).to(device)

        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Early Stopping
        best_acc = 0.0
        best_state = None
        patience_counter = 0

        # 6. Epoch Loop
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            t_loss, t_correct, t_total = 0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = loss_fn(out, by)
                loss.backward()
                optimizer.step()
                
                t_loss += loss.item()
                t_correct += (out.argmax(1) == by).sum().item()
                t_total += by.size(0)

            train_acc = (t_correct / t_total) * 100
            avg_loss = t_loss / len(train_loader)

            # Valid
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    v_correct += (out.argmax(1) == by).sum().item()
                    v_total += by.size(0)
            
            val_acc = (v_correct / v_total) * 100 if v_total > 0 else 0

            print(f"Ep {epoch:03d} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%", end="")

            # Check Best
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(" *")
            else:
                patience_counter += 1
                print(f" (Patience: {patience_counter})")

            if patience_counter >= args.patience:
                print(f"Early Stopping. Best Acc: {best_acc:.2f}%")
                break
        
        if best_state:
            save_name = f"{args.model_type}_{src}.pth"
            save_path = os.path.join(save_root, save_name)
            torch.save(best_state, save_path)
            print(f"Saved Model: {save_path}")

if __name__ == "__main__":
    main()
