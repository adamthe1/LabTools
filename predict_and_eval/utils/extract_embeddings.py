import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Fix for Qt platform plugin error - must be before importing matplotlib
import pandas as pd
from tqdm import tqdm
from deep_learning.preprocessing import args
import time
from pheno_data.get_subj_data import get_labels

from deep_learning.model import motionBert_rope, motionBert_4d
from deep_learning.preprocessing import preprocessing
from deep_learning.training.utils.training_helper import load_checkpoint
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, device='cuda', amount_device=2):
    """
    Load the trained MotionBERT model from a checkpoint file.
    """
    # Initialize the model
    if "54box8yt" in model_path:
        model_backbone = motionBert_rope.DSTformer(num_joints=26, maxlen=256 * 5,
                                                   use_rope=True)
        model = motionBert_rope.ReconstructNet(model_backbone, dim_in=4)
    else:
        model_backbone = motionBert_4d.DSTformer(num_joints=26, maxlen=256,
                                                 use_rope=False)
        model = motionBert_4d.ReconstructNet(model_backbone, dim_in=4)

    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    raw_sd = checkpoint['model']

    # First, strip any DataParallel
    if any(k.startswith('module.') for k in raw_sd):
        print("Stripping DataParallel prefix")
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}

    # Next, strip any TorchDynamo
    if any(k.startswith('_orig_mod.') for k in raw_sd):
        print("Stripping Dynamo wrapper prefix")
        raw_sd = {k[len('_orig_mod.'):]: v for k, v in raw_sd.items()}

    # raw_sd now has clean keys like 'model_backbone.temp_embed', etc.
    print("Final state_dict keys sample:", list(raw_sd.keys())[:5])

    model.load_state_dict(raw_sd, strict=False)
    if amount_device > 1:
        model = nn.DataParallel(model)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


def extract_embeddings(model, data_loader, device='cuda', amount_device=1):
    """
    Extract embeddings from the model for all sequences in the data loader.
    Mean-pool over time and joints to get a single embedding per sequence.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_activities = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, skeleton_data in tqdm(enumerate(data_loader), total=len(data_loader),
                                             desc="Extracting embeddings"):
            data = skeleton_data['data'].to(device)
            labels = skeleton_data['label_list']
            ids = skeleton_data['id']
            if 'activity' in skeleton_data:
                activities = skeleton_data['activity']
            else:
                activities = ['all'] * len(ids)

            # Extract embeddings from the model
            if isinstance(model, nn.DataParallel) and amount_device == 1:
                # when the model was wrapped in DataParallel but we are using only one GPU
                embeddings = model.module.get_representation(data)
            else:
                # when using DataParallel with multiple GPUs we want to use the model directly
                embeddings = model(data, representation=True)

            # embeddings: [B, seq_len, num_joints, dim_feat]
            B, F, J, D = embeddings.shape

            # Flatten time & joints into one dimension for simple pooling:
            flat = embeddings.view(B, F * J, D)  # [B, F*J, D]

            # Mean?pool
            mean_pool = flat.mean(dim=1)  # [B, D]

            # Max?pool
            max_pool = flat.max(dim=1).values  # [B, D]

            # Concat into a 2D vector
            embeddings = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2*D]

            all_embeddings.append(embeddings.cpu().numpy())
            for i, label_list in enumerate(labels):
                if i >= len(all_labels):
                    all_labels.append([])
                all_labels[i].extend(label_list.tolist())
            all_activities.extend(activities)
            all_ids.extend(ids)

    return np.vstack(all_embeddings), all_labels, all_activities, all_ids


# Create better structured DataFrames with proper column names
def create_embedding_dataframe(embeddings, labels, label_names, activities, ids):
    # Create a dictionary to hold all data
    data = {}

    # Add embedding columns with proper names
    for i in range(embeddings.shape[1]):
        data[f'embedding_{i}'] = embeddings[:, i]

    # Add labels with proper names
    for i, label_list in enumerate(labels):
        data[label_names[i]] = label_list

    # Add activity and ID columns
    data['activity'] = activities
    data['subject_id'] = ids

    # Create DataFrame with all columns at once to avoid fragmentation
    return pd.DataFrame(data).set_index('subject_id')


def main(
        model_path="/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/models/motionBert/bert_54box8yt_05_08_12_29/epoch_31.pth",
        save_dir="/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/motionBert/retest_31_epoch_1_seq"):
    # /net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/models/motionBert/bert_b39i2cm2_05_06_00_28/epoch_8.pth
    do_train = True
    do_eval = True
    do_test = True
    # turn off masking and augmentation
    args_cfg = args.default_args()
    args_cfg.random_mask = 0
    args_cfg.random_mask_frames = 0
    args_cfg.one_seq_per_person = True
    args_cfg.augments = {'None': True, 'rotation': False, 'scaling': False, 'translation': False, 'jittering': False,
                'mirroring': False,
                'temporal_downsample_slicing0': False, 'temporal_downsample_slicing1': False,
                'temporal_downsample_interpole': False,
                'temporal_upsample': False}
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    amount_device = 1

    # Define labels and task types (need these for the dataset creation)
    labels = ["age", "gender", "bmi", "exercise_score", "nerves_anxiety_tension_depression_doctor",
              "Attention Deficit Disorder (ADHD)", 'AHI', "total_scan_vat_mass", "sitting_blood_pressure_pulse_rate"]

    # 1. Load model
    model = load_model(model_path, device, amount_device=amount_device)
    if "54box8yt" in model_path and False:
        # 2. Create data loaders
        train_dataset, test_dataset, eval_dataset = preprocessing_five_seq.get_datasets(
            size_seq=256,
            labels=["age"],
            graph_data=False,
            overlap_sequence=0,
            args_cfg=args_cfg
        )
    else:
        train_dataset, test_dataset, eval_dataset = preprocessing.get_datasets(
            size_seq=256,
            labels=["age"],
            graph_data=False,
            overlap_sequence=0,
            args_cfg=args_cfg
        )

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=6, shuffle=False, num_workers=0)
    test_loader = DataLoader(eval_dataset, batch_size=6, shuffle=False, num_workers=0)
    print(
        f"Data loaders created. Train dataset: {len(train_dataset)} samples, Test dataset: {len(eval_dataset)} samples")

    # 3. Extract embeddings
    print("Extracting embeddings...")
    if do_train:
        E_train, train_labels, train_activities, train_ids = extract_embeddings(model, train_loader, device, amount_device=amount_device)
        train_data = {}
        for i in range(E_train.shape[1]):
            train_data[f'embedding_{i}'] = E_train[:, i]
        train_data['activity'] = train_activities
        train_data['subject_id'] = train_ids
        train_df = pd.DataFrame(train_data).set_index('subject_id')
        train_df.to_csv(os.path.join(save_dir, "train_embeddings.csv"))
        print(f"Extracted embeddings - Train: {E_train.shape}")
    if do_eval:
        E_eval, eval_labels, eval_activities, eval_ids = extract_embeddings(model, eval_loader, device, amount_device=amount_device)
        eval_data = {}

        for i in range(E_eval.shape[1]):
            eval_data[f'embedding_{i}'] = E_eval[:, i]

        eval_data['activity'] = eval_activities
        eval_data['subject_id'] = eval_ids

        eval_df = pd.DataFrame(eval_data).set_index('subject_id')

        eval_df.to_csv(os.path.join(save_dir, "eval_embeddings.csv"))
        print(f"Extracted embeddings - Eval: {E_eval.shape}")
    if do_test:
        E_test, test_labels, test_activities, test_ids = extract_embeddings(model, test_loader, device,
                                                                            amount_device=amount_device)
        test_data = {}

        for i in range(E_test.shape[1]):
            test_data[f'embedding_{i}'] = E_test[:, i]

        test_data['activity'] = test_activities
        test_data['subject_id'] = test_ids

        test_df = pd.DataFrame(test_data).set_index('subject_id')

        test_df.to_csv(os.path.join(save_dir, "test_embeddings.csv"))
        print(f"Extracted embeddings - Test: {E_test.shape}")


    with open(os.path.join(save_dir, "model_path.txt"), "w") as f:
        f.write(model_path)

    print(f"Saved embeddings to {save_dir}")



if __name__ == "__main__":
    main()