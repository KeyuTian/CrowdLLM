import ast
import numpy as np
import pandas as pd
import torch
import torch.utils.data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, config, ohe_feature_names):
        self.df = dataframe
        self.config = config
        self.ohe_feature_names = ohe_feature_names
        self.qids = self.df['qid'].values
        self.uids = self.df['uid'].values
        self.contexts = torch.tensor(self.df[self.ohe_feature_names].values, dtype=torch.float32)
        self.task_embeddings = torch.stack([torch.tensor(emb) for emb in self.df['task_embedding'].values])
        self.human_scores = torch.tensor(self.df['human_score'].values, dtype=torch.float32)
        self.yref_llm = torch.tensor(self.df['yref_llm'].values, dtype=torch.float32)
        self.baseline_llm_ratings_dist = torch.tensor(self.df['baseline_llm_ratings_dist'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "qid": self.qids[idx],
            "uid": self.uids[idx],
            "context": self.contexts[idx],
            "task_embedding": self.task_embeddings[idx],
            "human_score": self.human_scores[idx],
            "yref_llm": self.yref_llm[idx],
            "baseline_llm_ratings_dist": self.baseline_llm_ratings_dist[idx]
        }

class CustomDatasetnew(torch.utils.data.Dataset):
    def __init__(self, dataframe, config, ohe_feature_names):
        self.df = dataframe
        self.config = config
        self.ohe_feature_names = ohe_feature_names
        self.qids = self.df['qid'].values
        self.uids = self.df['uid'].values
        array = self.df[self.ohe_feature_names].values.astype('float32')
        self.contexts = torch.tensor(array)
        self.df = self.df.copy()  
        self.df['task_embedding'] = self.df['task_embedding'].apply(ast.literal_eval)   
        self.task_embeddings = torch.stack(
            [torch.tensor(emb, dtype=torch.float32) for emb in self.df['task_embedding'].values])
        self.task_embeddings = self.task_embeddings.to(torch.float32)
        self.human_scores = torch.tensor(self.df['human_score'].values, dtype=torch.float32)
        self.yref_llm = torch.tensor(self.df['yref_llm'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "qid": self.qids[idx],
            "uid": self.uids[idx],
            "context": self.contexts[idx],
            "task_embedding": self.task_embeddings[idx],
            "human_score": self.human_scores[idx],
            "yref_llm": self.yref_llm[idx]
        }
def cell_to_float32_array(x):
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    raise TypeError(f"Unsupported cell type: {type(x)}")

class CustomDatasetnew_recom(torch.utils.data.Dataset):
    def __init__(self, dataframe, config, ohe_feature_names):
        self.df = dataframe
        self.config = config
        self.ohe_feature_names = ohe_feature_names
        self.qids = self.df['qid'].values
        self.uids = self.df['uid'].values
        vecs = self.df['comment'].apply(cell_to_float32_array)
        lens = vecs.map(len)
        if lens.nunique() > 1:
            target = lens.mode().iloc[0]
            bad = lens != target
            vecs = vecs[~bad]
        array = np.stack(vecs.to_list(), axis=0).astype(np.float32)
        self.contexts = torch.tensor(array)
        self.task_embeddings = torch.stack([torch.tensor(emb) for emb in self.df['task_embedding'].values])
        self.task_embeddings = self.task_embeddings.to(torch.float32)
        self.human_scores = torch.tensor(self.df['human_score'].values, dtype=torch.float32)
        self.yref_llm = torch.tensor(self.df['yref_llm'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "qid": self.qids[idx], 
            "uid": self.uids[idx],
            "context": self.contexts[idx],
            "task_embedding": self.task_embeddings[idx],
            "human_score": self.human_scores[idx],
            "yref_llm": self.yref_llm[idx]
        }
