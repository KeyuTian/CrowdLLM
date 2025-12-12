import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import CustomData as CD
import scipy.stats as st

def preprocess_data(df_main, df_llm_single_ref, df_llm_baseline_dist, config,frac= 1):
    print("Starting data preprocessing...")
    # Rename core columns for consistency
    df = df_main.rename(columns={
        config["qid_col"]: "qid",
        config["uid_col"]: "uid",
        config["human_score_col"]: "human_score",
        config["embedding_col"]: "task_embedding_str"
    }).drop(columns=[config["text_col"]], errors='ignore')
    # --- Context Feature Handling: One-Hot Encoding ---
    # Ensure context features are string type for consistent OHE
    for col in config["context_categorical_features"]:
        df[col] = df[col].astype(str)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # handle_unknown='ignore' is safer for test sets if new categories appear
    context_one_hot_encoded = ohe.fit_transform(df[config["context_categorical_features"]])
    ohe_feature_names = ohe.get_feature_names_out(config["context_categorical_features"])
    df_context_ohe = pd.DataFrame(context_one_hot_encoded, columns=ohe_feature_names, index=df.index)
    #print(df_context_ohe.shape)
    # Dynamically determine context_size
    dynamic_context_size = df_context_ohe.shape[1]
    print('context_size:',dynamic_context_size)

    print(f"Context features one-hot encoded. New context_size: {dynamic_context_size}")
    config['context_size'] = dynamic_context_size # Update global config

    # Combine OHE context with the rest of df
    df = pd.concat([df.drop(columns=config["context_categorical_features"]), df_context_ohe], axis=1)

    # --- Task Embedding ---
    # Convert string representation of embedding to numpy array
    # This is slow, but done once.
    print("Converting task embeddings from string to numpy array...")
    # df["task_embedding"] = df["task_embedding_str"].apply(ast.literal_eval)
    # print(type(kk[0]))
    df["task_embedding"] = df["task_embedding_str"].apply(lambda x: np.array(eval(x), dtype=np.float32))
    #print(type(df["task_embedding"][0]))
    df = df.drop(columns=["task_embedding_str"])

    # --- Merge LLM y_ref (from dataset_A) ---
    df_llm_single_ref = df_llm_single_ref.rename(columns={
        config["qid_col"]: "qid",
        config["llm_ref_score_col_dsA"]: "yref_llm"
    })
    df = df.merge(df_llm_single_ref[["qid", "yref_llm"]], on="qid", how="left")
    # Fill missing yref_llm, e.g., with a neutral value or mean. Using 0 for now.
    df["yref_llm"] = df["yref_llm"].fillna(0.0).astype(np.float32)

    # --- Merge LLM baseline distribution (from dataset_C) ---
    df_llm_baseline_dist = df_llm_baseline_dist.rename(columns={config["qid_col"]: "qid"})
    baseline_cols_dsC = [f"{config['baseline_llm_prefix_dsC']}{i}" for i in range(1, config["num_baseline_llm_ratings"] + 1)]

    # Check if all baseline columns exist
    missing_baseline_cols = [col for col in baseline_cols_dsC if col not in df_llm_baseline_dist.columns]
    if missing_baseline_cols:
        print(f"Warning: Missing baseline LLM columns in dataset_C: {missing_baseline_cols}")
        # Create missing columns with NaN if they don't exist
        for col in missing_baseline_cols:
            df_llm_baseline_dist[col] = np.nan

    df_llm_baseline_dist["baseline_llm_ratings_dist"] = df_llm_baseline_dist[baseline_cols_dsC].values.tolist()
    df_llm_baseline_dist["baseline_llm_ratings_dist"] = df_llm_baseline_dist["baseline_llm_ratings_dist"].apply(
        lambda x: [float(val) if pd.notna(val) else 0.0 for val in x] # Handle NaNs within lists, fill with 0
    )
    df = df.merge(df_llm_baseline_dist[["qid", "baseline_llm_ratings_dist"]], on="qid", how="left")
    # Fill missing lists with a default list of zeros
    default_dist = [0.0] * config["num_baseline_llm_ratings"]
    df["baseline_llm_ratings_dist"] = df["baseline_llm_ratings_dist"].apply(lambda x: x if isinstance(x, list) else default_dist)

    # Ensure human_score is float
    df["human_score"] = df["human_score"].astype(np.float32)

    # --- Determine dynamic score range ---
    if not df["human_score"].empty:
        actual_min_score = df["human_score"].min()
        actual_max_score = df["human_score"].max()
        config["dynamic_score_min"] = float(actual_min_score)
        config["dynamic_score_max"] = float(actual_max_score)
        print(f"Determined dynamic score range from 'human_score': Min={config['dynamic_score_min']}, Max={config['dynamic_score_max']}")
    else:
        # Fallback if human_score column is empty or all NaN, though this should ideally not happen for training data
        print("Warning: 'human_score' column is empty or all NaN. Using default score range (1-10) for clamping.")
        config["dynamic_score_min"] = 1.0
        config["dynamic_score_max"] = 10.0

    # Select final columns (uid might be useful for some analyses but not directly for model input here)
    final_cols = ["qid", "uid", "human_score", "task_embedding", "yref_llm", "baseline_llm_ratings_dist"] + list(ohe_feature_names)
    df = df[final_cols]

    print(f"Preprocessing complete. Final DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"NaN check:\n{df.isnull().sum()}")

    unique_qids_overall = df['qid'].unique()
    global_train_val_qids, global_test_qids = train_test_split(
        unique_qids_overall,
        test_size=config["global_test_split_size"],
        random_state=config["seed"],
        shuffle=True
    )
    global_train_val_df = df[df['qid'].isin(global_train_val_qids)].copy()
    global_test_df = df[df['qid'].isin(global_test_qids)].copy()
    print(f"Global Train/Val set: {len(global_train_val_df)} rows, {len(global_train_val_qids)} QIDs")
    print(f"Global Test set: {len(global_test_df)} rows, {len(global_test_qids)} QIDs")
    global_train_val_df.reset_index(drop = True)
    global_test_df.reset_index(drop = True)
    if frac < 1.0:
        all_uids = global_train_val_df['uid'].unique()
        keep_uids, _ = train_test_split(
            all_uids,
            test_size=1 - frac,
            random_state=config["seed"],
            shuffle=True
        )
        sub_train_df = global_train_val_df[
            global_train_val_df['uid'].isin(keep_uids)
        ].reset_index(drop=True)
    else:
        sub_train_df = global_train_val_df.copy()
    print(f"[Info!] Training subset: {sub_train_df['uid'].nunique()} unique users, {len(sub_train_df)} rows")
    print(f"[Info!] Testing set:", global_test_df.head(3))
    return sub_train_df, global_test_df, ohe_feature_names # Return OHE feature names for CustomDataset

def split_train_test(df, config, frac = 1.0):
    print("Starting data preprocessing...")
    # --- Determine dynamic score range ---
    if not df["human_score"].empty:
        actual_min_score = df["human_score"].min()
        actual_max_score = df["human_score"].max()
        config["dynamic_score_min"] = float(actual_min_score)
        config["dynamic_score_max"] = float(actual_max_score)
        print(f"Determined dynamic score range from 'human_score': Min={config['dynamic_score_min']}, Max={config['dynamic_score_max']}")
    else:
        # Fallback if human_score column is empty or all NaN, though this should ideally not happen for training data
        print("Warning: 'human_score' column is empty or all NaN. Using default score range (1-10) for clamping.")
        config["dynamic_score_min"] = 1.0
        config["dynamic_score_max"] = 10.0

    unique_qids_overall = df['qid'].unique()
    global_train_val_qids, global_test_qids = train_test_split(
        unique_qids_overall,
        test_size=config["global_test_split_size"],
        random_state=config["seed"],
        shuffle=True
    )
    global_train_val_df = df[df['qid'].isin(global_train_val_qids)].copy()
    global_test_df = df[df['qid'].isin(global_test_qids)].copy()
    print(f"Global Train/Val set: {len(global_train_val_qids)} QIDs,{global_train_val_df['uid'].nunique()} unique users, {len(global_train_val_df)} rows;")
    print(f"Global Test set: {len(global_test_qids)} QIDs, {global_test_df['uid'].nunique()} unique users, {len(global_test_df)} rows;")
    global_train_val_df.reset_index(drop = True)
    global_test_df.reset_index(drop = True)

    if frac < 1.0:
        all_uids = global_train_val_df['uid'].unique()
        keep_uids, _ = train_test_split(
            all_uids,
            test_size=1 - frac,
            random_state=config["seed"],
            shuffle=True
        )
        sub_train_df = global_train_val_df[
            global_train_val_df['uid'].isin(keep_uids)
        ].reset_index(drop=True)
    else:
        sub_train_df = global_train_val_df.copy()
    print(f"[Info!] Training subset: {sub_train_df['uid'].nunique()} unique users, {len(sub_train_df)} rows")
    print(f"[Info!] Testing set:", global_test_df.head(3))
    return sub_train_df, global_test_df

def split_train_test2(df, config, frac = 1.0):
    print("Starting data preprocessing...")
    if not df["human_score"].empty:
        actual_min_score = df["human_score"].min()
        actual_max_score = df["human_score"].max()
        config["dynamic_score_min"] = float(actual_min_score)
        config["dynamic_score_max"] = float(actual_max_score)
        print(f"Determined dynamic score range from 'human_score': Min={config['dynamic_score_min']}, Max={config['dynamic_score_max']}")
    else:
        # Fallback if human_score column is empty or all NaN, though this should ideally not happen for training data
        print("Warning: 'human_score' column is empty or all NaN. Using default score range (1-10) for clamping.")
        config["dynamic_score_min"] = 1.0
        config["dynamic_score_max"] = 10.0

    unique_qids_overall = df['uid'].unique()
    global_train_val_qids, global_test_qids = train_test_split(
        unique_qids_overall,
        test_size=config["global_test_split_size"],
        random_state=config["seed"],
        shuffle=True
    )
    global_train_val_df = df[df['uid'].isin(global_train_val_qids)].copy()
    global_test_df = df[df['uid'].isin(global_test_qids)].copy()
    print(f"Global Train/Val set: {len(global_train_val_qids)} UIDs,{global_train_val_df['qid'].nunique()} unique questions, {len(global_train_val_df)} rows;")
    print(f"Global Test set: {len(global_test_qids)} UIDs, {global_test_df['qid'].nunique()} unique questions, {len(global_test_df)} rows;")
    global_train_val_df.reset_index(drop = True)
    global_test_df.reset_index(drop = True)

    if frac < 1.0:
        all_uids = global_train_val_df['uid'].unique()
        keep_uids, _ = train_test_split(
            all_uids,
            test_size=1 - frac,
            random_state=config["seed"],
            shuffle=True
        )
        sub_train_df = global_train_val_df[
            global_train_val_df['uid'].isin(keep_uids)
        ].reset_index(drop=True)
    else:
        sub_train_df = global_train_val_df.copy()
    print(f"[Info!] Training subset: {sub_train_df['uid'].nunique()} unique users, {len(sub_train_df)} rows")
    print(f"[Info!] Testing set:", global_test_df.head(3))
    return sub_train_df, global_test_df

def split_train_test3(df, config, frac = 1.0):
    print("Starting data preprocessing...")
    # --- Determine dynamic score range ---
    if not df["human_score"].empty:
        actual_min_score = df["human_score"].min()
        actual_max_score = df["human_score"].max()
        config["dynamic_score_min"] = float(actual_min_score)
        config["dynamic_score_max"] = float(actual_max_score)
        print(f"Determined dynamic score range from 'human_score': Min={config['dynamic_score_min']}, Max={config['dynamic_score_max']}")
    else:
        print("Warning: 'human_score' column is empty or all NaN. Using default score range (1-10) for clamping.")
        config["dynamic_score_min"] = 1.0
        config["dynamic_score_max"] = 10.0

    unique_qids_overall = df['qid'].unique() #todo
    global_train_val_qids, global_test_qids = train_test_split(
        unique_qids_overall,
        test_size=config["global_test_split_size"],
        random_state=config["seed"],
        shuffle=True
    )
    global_train_val_df = df[df['qid'].isin(global_train_val_qids)].copy()
    global_test_df = df[df['qid'].isin(global_test_qids)].copy()
    print(f"Global Train/Val set: {len(global_train_val_qids)} UIDs,{global_train_val_df['qid'].nunique()} unique questions, {len(global_train_val_df)} rows;")
    print(f"Global Test set: {len(global_test_qids)} UIDs, {global_test_df['qid'].nunique()} unique questions, {len(global_test_df)} rows;")
    global_train_val_df.reset_index(drop = True)
    global_test_df.reset_index(drop = True)

    if frac < 1.0:
        all_uids = global_train_val_df['uid'].unique()
        keep_uids, _ = train_test_split(
            all_uids,
            test_size=1 - frac,
            random_state=config["seed"],
            shuffle=True
        )
        sub_train_df = global_train_val_df[
            global_train_val_df['uid'].isin(keep_uids)
        ].reset_index(drop=True)
    else:
        sub_train_df = global_train_val_df.copy()
    print(f"[Info!] Training subset: {sub_train_df['uid'].nunique()} unique users, {len(sub_train_df)} rows")
    print(f"[Info!] Testing set:", global_test_df.head(3))
    return sub_train_df, global_test_df

def create_train_val_loaders(current_train_val_df, config, ohe_feature_names, for_reduction_experiment=False):
    """
    Splits current_train_val_df into train and validation sets and creates DataLoaders.
    The test set is global and handled separately.
    """
    if current_train_val_df.empty:
        print("Warning: current_train_val_df is empty. Cannot create train/val loaders.")
        return None, None, pd.DataFrame(columns=current_train_val_df.columns), pd.DataFrame(
            columns=current_train_val_df.columns)

    unique_qids_in_current_tv = current_train_val_df['qid'].unique()

    val_split_size = config["validation_split_from_train_size"]

    if len(unique_qids_in_current_tv) < 5 or val_split_size == 0:  # Heuristic for minimal val set
        actual_train_qids = unique_qids_in_current_tv
        val_qids = np.array([])
    else:
        actual_train_qids, val_qids = train_test_split(
            unique_qids_in_current_tv,
            test_size=val_split_size,
            random_state=config["seed"],  # Use global seed for this split too
            shuffle=True
        )

    train_df = current_train_val_df[current_train_val_df['qid'].isin(actual_train_qids)]
    val_df = current_train_val_df[current_train_val_df['qid'].isin(val_qids)] if len(val_qids) > 0 else pd.DataFrame(
        columns=current_train_val_df.columns)

    if len(train_df) == 0:
        print("  Warning: Training DataFrame is empty after split.")
        return None, None, train_df, val_df
    train_dataset = CD.CustomDatasetnew(train_df, config, ohe_feature_names)
    val_dataset = CD.CustomDatasetnew(val_df, config, ohe_feature_names) if len(val_df) > 0 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    ) if val_dataset else None

    return train_loader, val_loader, train_df, val_df

def create_train_val_loaders_recom(current_train_val_df, config, ohe_feature_names, for_reduction_experiment=False):
    """
    Splits current_train_val_df into train and validation sets and creates DataLoaders.
    The test set is global and handled separately.
    """
    if current_train_val_df.empty:
        print("Warning: current_train_val_df is empty. Cannot create train/val loaders.")
        return None, None, pd.DataFrame(columns=current_train_val_df.columns), pd.DataFrame(
            columns=current_train_val_df.columns)

    unique_qids_in_current_tv = current_train_val_df['qid'].unique()

    val_split_size = config["validation_split_from_train_size"]

    if len(unique_qids_in_current_tv) < 5 or val_split_size == 0:  # Heuristic for minimal val set
        actual_train_qids = unique_qids_in_current_tv
        val_qids = np.array([])
        # print("  Note: Using all current_train_val_df for training, no validation split.")
    else:
        actual_train_qids, val_qids = train_test_split(
            unique_qids_in_current_tv,
            test_size=val_split_size,
            random_state=config["seed"],  # Use global seed for this split too
            shuffle=True
        )

    train_df = current_train_val_df[current_train_val_df['qid'].isin(actual_train_qids)]
    val_df = current_train_val_df[current_train_val_df['qid'].isin(val_qids)] if len(val_qids) > 0 else pd.DataFrame(
        columns=current_train_val_df.columns)

    if len(train_df) == 0:
        print("  Warning: Training DataFrame is empty after split.")
        return None, None, train_df, val_df

    train_dataset = CD.CustomDatasetnew_recom(train_df, config, ohe_feature_names)
    val_dataset = CD.CustomDatasetnew_recom(val_df, config, ohe_feature_names) if len(val_df) > 0 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    ) if val_dataset else None

    return train_loader, val_loader, train_df, val_df

def cell_to_float32_array(x):
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    raise TypeError(f"Unsupported cell type: {type(x)}")

def cell_to_int64_array(x):
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.int64)
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.int64)
    raise TypeError(f"Unsupported cell type: {type(x)}")

def flatten_error_each_question(error_each_question, agent_keys):
    records_mae = []
    records_time = []
    for record in error_each_question:
        num_workers = record["num_workers"]
        for agent in agent_keys:
            if agent in record:
                for error_value in record[agent]:
                    records_mae.append({
                        "num_workers":num_workers,
                        "agent":agent,
                        "error":error_value
                    })
    for record in error_each_question:
        num_workers = record["num_workers"]
        records_time.append({
            "num_workers":num_workers,
            "time":record["time"]
        })

    return pd.DataFrame(records_mae), pd.DataFrame(records_time)


def compute_mae_ci_per_agent_long(df_long, ci_level=0.95):
    summary = []

    for (agent, num_workers), group in df_long.groupby(["agent", "num_workers"]):
        errors = group["error"].values
        mean = np.mean(errors)
        std = np.std(errors, ddof=1)
        n = len(errors)
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        summary.append({
            "agent":agent,
            "num_workers":num_workers,
            "mean":mean,
            "std":std,
            "ci":ci
        })

    return pd.DataFrame(summary)


def plot_from_long_summary(summary_df, agent_colors, agent_label_map):
    plt.figure(figsize=(8, 5))
    for agent, color in agent_colors.items():
        sub = summary_df[summary_df["agent"] == agent]
        x = sub["num_workers"].values
        y = sub["mean"].values
        ci = sub["ci"].values
        label = agent_label_map.get(agent, agent.replace("_", " ")) if agent_label_map else agent.replace("_", " ")
        plt.plot(x, y, label=label, color=color)
        plt.fill_between(x, y - ci, y + ci, alpha=0.1, color=color)

    plt.xlabel("Number of LLM Workers",fontsize=16)
    plt.ylabel("MAE",fontsize=16)
    plt.xlim(left=min(summary_df["num_workers"]))
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_time_comparison(df_crowd, df_llm):
    df_crowd = df_crowd.rename(columns={df_crowd.columns[1]:"CrowdLLM"})
    df_llm = df_llm.rename(columns={df_llm.columns[1]:"LLM"})

    df_merged = pd.merge(df_crowd, df_llm, on="num_workers", how="inner")

    plt.figure(figsize=(8, 5))
    plt.plot(df_merged["num_workers"], df_merged["CrowdLLM"],
             color="red", label="CrowdLLM", linewidth=2)
    plt.plot(df_merged["num_workers"], df_merged["LLM"],
             color="purple", label="LLM", linewidth=2)

    plt.xlabel("Number of Workers")
    plt.ylabel("Time (s)")
    #plt.title("Inference Time vs Number of Workers")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def safe_ratio_for_reduction(df,
                             mode="task",
                             min_workers_per_task=1,
                             min_tasks_per_worker=1,
                             check_points=50,
                             trials=20):
    all_tasks = df['qid'].unique()
    all_workers = df['uid'].unique()
    total_tasks = len(all_tasks)
    total_workers = len(all_workers)
    safe_ratio = 1.0

    for r in np.linspace(1.0, 0.1, check_points):
        all_good = True
        for _ in range(trials):
            if mode == "task":
                sampled_tasks = np.random.choice(df['qid'].unique(),
                                                 size=max(int(total_tasks * r), min_tasks_per_worker),
                                                 replace=False)
                reduced_df = df[df['qid'].isin(sampled_tasks)]

                worker_task_counts = reduced_df.groupby('uid')['qid'].nunique()
                valid_workers = all(worker_task_counts.get(w, 0) >= min_tasks_per_worker
                                    for w in all_workers)
                if not valid_workers:
                    all_good = False
                    break

            elif mode == "worker":
                sampled_workers = np.random.choice(df['uid'].unique(),
                                                   size=max(int(total_workers * r), min_workers_per_task),
                                                   replace=False)
                reduced_df = df[df['uid'].isin(sampled_workers)]

                task_worker_counts = reduced_df.groupby('qid')['uid'].nunique()
                valid_tasks = all(task_worker_counts.get(t, 0) >= min_workers_per_task
                                  for t in all_tasks)
                if not valid_tasks:
                    all_good = False
                    break

        if not all_good:
            break
        safe_ratio = r
    return safe_ratio

def ablation_with_safe_ratios(df, mode="worker",
                              min_workers_per_task=1, min_tasks_per_worker=1,
                              points=10, check_points=50):

    total_workers = df['uid'].nunique()
    total_tasks = df['qid'].nunique()
    safe_ratios = []
    if mode in ["task", "worker"]:
        safe_ratio = safe_ratio_for_reduction(df,
                                              mode=mode,
                                              min_workers_per_task=min_workers_per_task,
                                              min_tasks_per_worker=min_tasks_per_worker,
                                              check_points=check_points)
    else:
        safe_ratio = 0.1 

    total_workers = df['uid'].nunique()
    total_tasks = df['qid'].nunique()

    safe_ratios = np.linspace(1.0, safe_ratio, points)

    safe_ratios = np.linspace(max(safe_ratios), min(safe_ratios), points)

    sampled_results = {}
    stats_list = [] 

    for r in safe_ratios:
        if mode == "worker":
            sampled_workers = np.random.choice(df['uid'].unique(), size=int(total_workers * r), replace=False)
            reduced_df = df[df['uid'].isin(sampled_workers)]

        elif mode == "task":
            sampled_tasks = np.random.choice(df['qid'].unique(), size=int(total_tasks * r), replace=False)
            reduced_df = df[df['qid'].isin(sampled_tasks)]

        elif mode == "both":
            sampled_workers = np.random.choice(df['uid'].unique(), size=int(total_workers * r), replace=False)
            sampled_tasks = np.random.choice(df['qid'].unique(), size=int(total_tasks * r), replace=False)
            reduced_df = df[df['uid'].isin(sampled_workers) & df['qid'].isin(sampled_tasks)]

        sampled_results[round(r, 2)] = reduced_df.copy()
        n_qid = reduced_df['qid'].nunique()
        n_uid = reduced_df['uid'].nunique()
        avg_workers_per_qid = reduced_df.groupby('qid')['uid'].nunique().mean()
        avg_qids_per_uid = reduced_df.groupby('uid')['qid'].nunique().mean()
        n_obs = len(reduced_df)

        stats_list.append({
            "ratio":round(r, 2),
            "n_qid":n_qid,
            "n_uid":n_uid,
            "avg_workers_per_qid":avg_workers_per_qid,
            "avg_qids_per_uid":avg_qids_per_uid,
            "n_obs":n_obs
        })

    stats_df = pd.DataFrame(stats_list)

    return safe_ratios, sampled_results, stats_df


def ablation_fixed_entities(df,
                            mode="responses_per_task",
                            step=1,
                            min_keep=1):

    all_workers = set(df['uid'].unique())
    all_tasks = set(df['qid'].unique())

    results = {}
    stats_list = []

    if mode == "responses_per_task":
        group_key = 'qid'
        check_entity = 'uid'
        all_entities = all_tasks
    elif mode == "tasks_per_worker":
        group_key = 'uid'
        check_entity = 'qid'
        all_entities = all_workers
    else:
        raise ValueError("mode must be 'responses_per_task' or 'tasks_per_worker'")
    group_counts = df.groupby(group_key)[check_entity].nunique()
    safe_limit = int(group_counts.min()) 
    max_remove = max(safe_limit - min_keep, 0)

    for remove in range(0, max_remove + 1, step):
        reduced_df_list = []

        for key, group in df.groupby(group_key):
            keep_n = max(group[check_entity].nunique() - remove, min_keep)
            sampled_group = group.sample(n=min(keep_n, len(group)), replace=False)
            reduced_df_list.append(sampled_group)

        reduced_df = pd.concat(reduced_df_list)
        still_has_all_tasks = set(reduced_df['qid'].unique()) == all_tasks
        still_has_all_workers = set(reduced_df['uid'].unique()) == all_workers

        if not (still_has_all_tasks and still_has_all_workers):
            break

        results[remove] = reduced_df.copy()

        n_qid = reduced_df['qid'].nunique()
        n_uid = reduced_df['uid'].nunique()
        avg_workers_per_qid = reduced_df.groupby('qid')['uid'].nunique().mean() if n_qid > 0 else 0
        avg_qids_per_uid = reduced_df.groupby('uid')['qid'].nunique().mean() if n_uid > 0 else 0
        n_obs = len(reduced_df)

        stats_list.append({
            "mode":mode,
            "remove":remove,
            "n_qid":n_qid,
            "n_uid":n_uid,
            "avg_workers_per_qid":avg_workers_per_qid,
            "avg_qids_per_uid":avg_qids_per_uid,
            "n_obs":n_obs
        })

    stats_df = pd.DataFrame(stats_list)
    return results, stats_df


def generate_scenario(df, scenario_id):
    sampled_results = {}
    stats_list = []

    if scenario_id == "worker":
        worker_counts = [1,3,5,7,9,11,13,15,17,19,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
        total_workers = df['uid'].nunique()

        for count in worker_counts:
            keep_n = min(count, total_workers)
            sampled_workers = np.random.choice(df['uid'].unique(), size=keep_n, replace=False)
            reduced_df = df[df['uid'].isin(sampled_workers)]
            sampled_results[keep_n] = reduced_df.copy()

            stats_list.append({
                "workers": keep_n,
                "n_qid": reduced_df['qid'].nunique(),
                "n_uid": reduced_df['uid'].nunique(),
                "avg_workers_per_qid": reduced_df.groupby('qid')['uid'].nunique().mean(),
                "avg_qids_per_uid": reduced_df.groupby('uid')['qid'].nunique().mean(),
                "n_obs": len(reduced_df)
            })

    elif scenario_id == "task":
        task_counts = [1,3,5,7,9,11,13,15,17,19,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
        total_tasks = df['qid'].nunique()

        for count in task_counts:
            keep_n = min(count, total_tasks)
            sampled_tasks = np.random.choice(df['qid'].unique(), size=keep_n, replace=False)
            reduced_df = df[df['qid'].isin(sampled_tasks)]
            sampled_results[keep_n] = reduced_df.copy()

            stats_list.append({
                "tasks": keep_n,
                "n_qid": reduced_df['qid'].nunique(),
                "n_uid": reduced_df['uid'].nunique(),
                "avg_workers_per_qid": reduced_df.groupby('qid')['uid'].nunique().mean(),
                "avg_qids_per_uid": reduced_df.groupby('uid')['qid'].nunique().mean(),
                "n_obs": len(reduced_df)
            })

    elif scenario_id == "tasks_per_worker":
        max_tasks = 20
        for t in range(1, max_tasks+1):
            reduced_df_list = []
            for uid, group in df.groupby('uid'):
                keep_n = min(t, len(group))
                sampled_group = group.sample(n=keep_n, replace=False)
                reduced_df_list.append(sampled_group)
            reduced_df = pd.concat(reduced_df_list)

            sampled_results[t] = reduced_df.copy()
            stats_list.append({
                "tasks_per_worker": t,
                "n_qid": reduced_df['qid'].nunique(),
                "n_uid": reduced_df['uid'].nunique(),
                "avg_workers_per_qid": reduced_df.groupby('qid')['uid'].nunique().mean(),
                "avg_qids_per_uid": reduced_df.groupby('uid')['qid'].nunique().mean(),
                "n_obs": len(reduced_df)
            })

    elif scenario_id == "responses_per_task":
        # 情景 5：workers_per_task 从 1 到 7
        max_workers = 7
        for w in range(1, max_workers+1):
            reduced_df_list = []
            for qid, group in df.groupby('qid'):
                keep_n = min(w, len(group))
                sampled_group = group.sample(n=keep_n, replace=False)
                reduced_df_list.append(sampled_group)
            reduced_df = pd.concat(reduced_df_list)

            sampled_results[w] = reduced_df.copy()
            stats_list.append({
                "workers_per_task": w,
                "n_qid": reduced_df['qid'].nunique(),
                "n_uid": reduced_df['uid'].nunique(),
                "avg_workers_per_qid": reduced_df.groupby('qid')['uid'].nunique().mean(),
                "avg_qids_per_uid": reduced_df.groupby('uid')['qid'].nunique().mean(),
                "n_obs": len(reduced_df)
            })
    else:
        raise ValueError("scenario_id must be 1, 2, 4, or 5")
    stats_df = pd.DataFrame(stats_list)
    return sampled_results, stats_df
def compute_ci(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = st.sem(data)  
    h = sem * st.t.ppf((1 + confidence) / 2., n-1)  
    return mean, h


def determine_fixed_k(runs_k):
    lengths = [len(k) for k in runs_k]
    avg_len = round(np.mean(lengths))
    k_start = max(k[0] for k in runs_k)
    k_end = min(k[-1] for k in runs_k)
    return np.linspace(k_start, k_end, avg_len)

def plot_with_confidence(k_values,
                         model1_mean, model1_ci,
                         model2_mean, model2_ci,title,
                         model1_label="Pure Generative",
                         model2_label="CrowdLLM",
                         ylabel="WD of each quest"
                         ):
    plt.figure(figsize=(8, 5))

    # Model 1
    plt.plot(k_values, model1_mean, label=model1_label, color="blue")
    plt.fill_between(k_values,
                     np.array(model1_mean) - np.array(model1_ci),
                     np.array(model1_mean) + np.array(model1_ci),
                     color="blue", alpha=0.2)

    # Model 2
    plt.plot(k_values, model2_mean, label=model2_label, color="red")
    plt.fill_between(k_values,
                     np.array(model2_mean) - np.array(model2_ci),
                     np.array(model2_mean) + np.array(model2_ci),
                     color="red", alpha=0.2)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
