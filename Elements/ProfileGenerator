import numpy as np
import torch
import torch.utils.data
from torch import nn
import random

class ProfileGenerator(nn.Module):
    def __init__(self, CONFIG, global_train_val_df, ohe_feature_names):
        super(ProfileGenerator, self).__init__()
        self.context_df_for_profiler_fit = global_train_val_df[ohe_feature_names]
        self.ohe_feature_names = list(ohe_feature_names)
        self.feature_probabilities = None  # For OHE features
        self.num_ohe_features = 0
        self.original_context_feature_specs = {}  # To store info about original features if needed later
        self._is_fitted = False  # Add a flag
        self.context_size = len(ohe_feature_names)
        self.device = CONFIG['device']
        self.fit()

    def fit(self):
        """
        Learns the characteristics of the one-hot encoded context features.
        context_df_ohe: A DataFrame of one-hot encoded context features from training data.
        ohe_feature_names_list: List of the OHE feature column names.
        """
        self.num_ohe_features = len(self.ohe_feature_names)

        if self.num_ohe_features == 0:
            print("Warning: No OHE features to learn for ProfileGenerator.")
            return

        # Calculate the probability of each OHE feature being 1
        # This assumes that for each original categorical variable, only one of its OHE columns can be 1.
        # print(self.context_df_for_profiler_fit)
        self.feature_probabilities = torch.tensor(self.context_df_for_profiler_fit.mean().values, dtype=torch.float32)
        print(f"ProfileGenerator fitted. Learned probabilities for {self.num_ohe_features} OHE features.")

        # To make generation more realistic (ensure one-hot constraint per original feature group):
        # We need to know which OHE columns belong to which original feature.
        # This requires parsing ohe_feature_names_list (e.g., "gender_Man", "gender_Woman").
        self.original_feature_groups = {}  # e.g., {"gender": ["gender_Man", "gender_Woman", ...]}
        original_feature_names_from_ohe = sorted(list(set([name.split('_')[0] for name in self.ohe_feature_names])))

        for orig_feature in original_feature_names_from_ohe: # orig_feature  = gender
            self.original_feature_groups[orig_feature] = [
                ohe_col for ohe_col in self.ohe_feature_names if ohe_col.startswith(orig_feature + "_")
            ]
            # Store probabilities for this group
            group_probs = [self.feature_probabilities[self.ohe_feature_names.index(ohe_col)].item()
                           for ohe_col in self.original_feature_groups[orig_feature]]
            self.original_context_feature_specs[orig_feature] = {
                "type":"categorical_ohe_group",
                "ohe_columns":self.original_feature_groups[orig_feature],
                "probabilities":torch.tensor(group_probs, dtype=torch.float32) / sum(group_probs) if sum(
                    group_probs) > 0 else torch.ones(len(group_probs)) / len(group_probs)  # Normalize
            }

        if self.num_ohe_features > 0:  # Consider it fitted if it processed some features
            self._is_fitted = True
        else:
            self._is_fitted = False

    def is_fitted(self):
        return self._is_fitted

    def generate_synthetic_profile(self,random_generate = False, temperature=1.0):
        """
        Generates a synthetic one-hot encoded context vector.
        Sampling is done via softmax-adjusted multinomial with a controllable temperature.
        """
        if self.num_ohe_features == 0 or self.feature_probabilities is None:
            if self.num_ohe_features > 0:
                return torch.zeros(self.num_ohe_features, device=self.device)
            else:
                raise ValueError("ProfileGenerator must be fitted before generating profiles.")

        synthetic_profile_ohe = torch.zeros(self.num_ohe_features, dtype=torch.float32)

        for orig_feature, spec in self.original_context_feature_specs.items():
            if spec["type"] == "categorical_ohe_group":
                raw_probs = spec["probabilities"]  # Tensor of shape [num_options]

                if not random_generate:
                    if temperature != 1.0:
                        adjusted_probs = torch.softmax(raw_probs / temperature, dim=0)
                    else:
                        adjusted_probs = raw_probs / raw_probs.sum()  # fallback to normalized

                    chosen_idx_in_group = torch.multinomial(adjusted_probs, 1).item()
                else:
                    chosen_idx_in_group = torch.randint(0, len(spec["ohe_columns"]), (1,)).item()
                chosen_ohe_col_name = spec["ohe_columns"][chosen_idx_in_group]

                global_idx = self.ohe_feature_names.index(chosen_ohe_col_name)
                synthetic_profile_ohe[global_idx] = 1.0
        #print('synthetic_profile_ohe:',synthetic_profile_ohe, synthetic_profile_ohe.type())
        return synthetic_profile_ohe.to(self.device)

    def generate_random_preferences(self):

        districts = ["District_Nord", "District_Ost", "District_S", "District_West"]


        # 三个正整数和为100
        t, c = sorted(random.sample(range(1, 99), 2))
        transport, culture, nature = t, c - t, 100 - c

        # 四个因素打分
        how_connected = random.randint(0, 4)
        f_district = random.randint(0, 4)
        f_topic = random.randint(0, 4)
        f_cost = random.randint(0, 4)
        f_likelihood = random.randint(0, 4)

        # District 独热编码（随机一个为 True）
        selected = random.choice(districts)
        district_bools = [selected == d for d in districts]

        full_list = [
            transport, culture, nature,
            how_connected, f_district, f_topic, f_cost, f_likelihood,
            *district_bools
        ]
        for i, x in enumerate(full_list):
            print(f"Index {i}: Value = {x}, Type = {type(x)}")
        synthetic_profile_ohe = torch.tensor(full_list, dtype=torch.int64)
        return synthetic_profile_ohe.to(self.device)

class ProfileGenerator_recom(nn.Module):
    def __init__(self, CONFIG, global_train_val_df, ohe_feature_names):
        super(ProfileGenerator_recom, self).__init__()
        self.context_df_for_profiler_fit = global_train_val_df[ohe_feature_names]
        self.ohe_feature_names = list(ohe_feature_names)
        self.feature_probabilities = None  # For OHE features
        self.num_ohe_features = 0
        self.original_context_feature_specs = {}  # To store info about original features if needed later
        self._is_fitted = False  # Add a flag
        self.context_size = len(ohe_feature_names)
        self.device = CONFIG['device']

    def generate_synthetic_profile(self,random_generate = False, temperature=1.0):
        EMBEDDING_DIM = 768
        comment = np.random.rand(EMBEDDING_DIM).tolist()
        synthetic_profile_ohe = torch.tensor(comment, dtype=torch.int64)

        return synthetic_profile_ohe.to(self.device)
