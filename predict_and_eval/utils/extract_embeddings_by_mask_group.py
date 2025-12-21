import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Fix for Qt platform plugin error - must be before importing matplotlib
import pandas as pd
from tqdm import tqdm
from deep_learning.preprocessing import args

import deep_learning.preprocessing.joints_file as joints_file

from deep_learning.model import motionBert_rope, motionBert_4d
from deep_learning.preprocessing import preprocessing


import warnings
from utils.visualyze_skeleton import plot_skeleton_with_bones

warnings.filterwarnings("ignore")


def load_model(model_path, device, amount_device=2):
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


def extract_embeddings(model, data_loader, device, amount_device=1, joint_keep_indices=None):
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
        counter = 1
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
                joints, embeddings = model(data, representation=False)

            # send the first joint from the batch to CPU for visualization and turn B, F, J to F, J, C where C is the number of coords
            if counter == 0:
                visualyze_joint = joints[0].cpu().numpy()  # [F, J, C]

                plot_skeleton_with_bones(
                    visualyze_joint, frame_idx=0, title=f"Skeleton for batch {batch_idx}, ID: {ids[0]}"
                )
                counter = 1
            # embeddings: [B, seq_len, num_joints, dim_feat]
            B, F, J, D = embeddings.shape

            if joint_keep_indices is not None:
                # create a mask to keep only the specified joints
                joint_keep_mask = torch.tensor(joint_keep_indices, device=device)
                # Apply the mask to embeddings
                embeddings = embeddings[:, :, joint_keep_mask, :]  # [B, F, num_joints_to_keep, D]
                J = len(joint_keep_indices)  # Update J to the number of kept joints

            # Flatten time & joints into one dimension for simple pooling:
            flat = embeddings.view(B, F * J, D)  # [B, F*J, D]

            # Mean and Max pool over the flattened dimension
            mean_embeddings = flat.mean(dim=1)  # [B, num_pairs, D]
            max_embeddings, _ = flat.max(dim=1)  # [B, num_pairs, D]

            # Concatenate mean and max embeddings
            embeddings = torch.cat([mean_embeddings, max_embeddings], dim=-1)

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

def extract_per_mask():
    """
    using noise joints, extract embeddings per mask.
    :return:
    """
    args_cfg = change_args() # change the args to use the noise joints
    return args_cfg
    joints_groups = joints_file.noise_pairs
    groups_to_add = {"legs": joints_groups['hips'] + joints_groups['knees'] +
                                joints_groups['ankles'] + joints_groups['feet'],
                     "arms": joints_groups['shoulders'] + joints_groups['elbows'] +
                                joints_groups['wrists'] + joints_groups['clavicles'],
                     "torso": (joints_groups['pelvis'], joints_groups['spine_navel'],
                                joints_groups['spine_chest'], joints_groups['neck']),
                     "full_head": ([joints_groups['head']] + [joints_groups['nose']] +
                                   list(joints_groups['eyes']) + list(joints_groups['ears']))}
    model_path = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/models/motionBert/bert_b39i2cm2_05_06_00_28/epoch_8.pth"
    dir_to_save = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/motionBert/with_masking_b39i2cm2/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amount_device = 1
    model = load_model(model_path, device, amount_device=amount_device)
    # add the groups to the joints_groups
    for group_name, joints in groups_to_add.items():
        joints_groups[group_name] = tuple(joints)
    # save the groups in a file
    with open(os.path.join(dir_to_save, "joints_groups.txt"), "w") as f:
        save_dict = {}
        for group_name, joints in joints_groups.items():
            if not isinstance(joints, tuple):
                joints = (joints,)
            save_dict[group_name] = joints
        f.write(str(save_dict))


    # create a keep mask with the groups
    all_joints = list(range(26))  # 26 joints in total
    print(joints_groups)
    for group_name, joints in joints_groups.items():
        if not isinstance(joints, tuple):
            joints = (joints,)
        # create a mask for the group
        joint_keep_mask = [j for j in all_joints if j not in joints]
        args.specific_joint_keep_mask = joint_keep_mask
        save_dir = os.path.join(dir_to_save, group_name)
        if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"{group_name} Directory already exists and has files.")
            continue
        os.makedirs(save_dir, exist_ok=True)
        print(f"Extracting embeddings for group {group_name} with joints {joint_keep_mask}")
        main(model=model, model_path=model_path, save_dir=save_dir, joints_keep_mask=joint_keep_mask)

def change_args():
    args_cfg = args.default_args()
    args_cfg.random_mask = 0
    args_cfg.random_mask_frames = 0
    args_cfg.augments = {'None': True, 'rotation': False, 'scaling': False, 'translation': False, 'jittering': False,
                     'mirroring': False,
                     'temporal_downsample_slicing0': False, 'temporal_downsample_slicing1': False,
                     'temporal_downsample_interpole': False,
                     'temporal_upsample': False}
    args_cfg.one_seq_per_person = False  # we want to extract embeddings per subject, not per sequence
    # add specific joint keep mask
    return args_cfg

def main(
        model,
        model_path="",
        save_dir="/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/motionBert/extracted_joint_level/",
        joints_keep_mask=None):
    #

    do_train = True
    do_eval = True
    do_test = True
    # turn off masking and augmentation
    size_seq = 256

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    amount_device = 1
    do_five_seq = False

    # Define labels and task types (need these for the dataset creation)
    labels = ["age", "gender", "bmi", "exercise_score", "nerves_anxiety_tension_depression_doctor",
              "Attention Deficit Disorder (ADHD)", 'AHI', "total_scan_vat_mass", "sitting_blood_pressure_pulse_rate"]

    # 1. Load model
    if "54box8yt" in model_path and do_five_seq:
        # 2. Create data loaders
        train_dataset, test_dataset, eval_dataset = preprocessing_five_seq.get_datasets(
            size_seq=256,
            labels=["age"],
            graph_data=False,
            overlap_sequence=0
        )
    else:
        train_dataset, test_dataset, eval_dataset = preprocessing.get_datasets(
            size_seq=256,
            labels=["age"],
            graph_data=False,
            overlap_sequence=0
        )
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
    print(
        f"Data loaders created. Train dataset: {len(train_dataset)} samples, Test dataset: {len(eval_dataset)} samples")

    # 3. Extract embeddings
    print("Extracting embeddings...")
    if do_train:
        E_train, train_labels, train_activities, train_ids = extract_embeddings(model, train_loader, device,
                                                                                amount_device=amount_device,
                                                                                joint_keep_indices=joints_keep_mask)
        train_data = {}
        for i in range(E_train.shape[1]):
            train_data[f'embedding_{i}'] = E_train[:, i]
        train_data['activity'] = train_activities
        train_data['subject_id'] = train_ids
        train_df = pd.DataFrame(train_data).set_index('subject_id')
        train_df.to_csv(os.path.join(save_dir, "train_embeddings.csv"))
        print(f"Extracted embeddings - Train: {E_train.shape}")
    if do_eval:
        E_eval, eval_labels, eval_activities, eval_ids = extract_embeddings(model, eval_loader, device,
                                                                            amount_device=amount_device,
                                                                            joint_keep_indices=joints_keep_mask)
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
                                                                            amount_device=amount_device,
                                                                            joint_keep_indices=joints_keep_mask)
        test_data = {}

        for i in range(E_test.shape[1]):
            test_data[f'embedding_{i}'] = E_test[:, i]

        test_data['activity'] = test_activities
        test_data['subject_id'] = test_ids

        test_df = pd.DataFrame(test_data).set_index('subject_id')

        test_df.to_csv(os.path.join(save_dir, "test_embeddings.csv"))
        print(f"Extracted embeddings - Test: {E_test.shape}")

    with open(os.path.join(save_dir, "model_path.txt"), "w") as f:
        f.write(f"{model_path}\n")
        f.write(f"size_seq: {size_seq}\n")

    print(f"Saved embeddings to {save_dir}")


if __name__ == "__main__":
    extract_per_mask()