import torch
import torch.utils.data
from torch import optim
import ProfileGenerator as PG
import BeliefGenerator as BG
import Blender
from Functions import split_train_test

class CrowdLLM_worker():
    def __init__(self, CONFIG, data, demo_name, multi = 1.0):
        self.CONFIG = CONFIG
        self.device = torch.device(self.CONFIG["device"])
        self.global_train_val_df, self.global_test_df = split_train_test(data, self.CONFIG)
        self.ohe_feature_names = demo_name
        self.CONFIG["context_size"] = len(demo_name)
        self.profile_generator = PG.ProfileGenerator(self.CONFIG, self.global_train_val_df, self.ohe_feature_names)
        self.profile = self.profile_generator.generate_synthetic_profile()
        self.belief_generator = BG.Generator(feature_size=self.CONFIG["feature_size"],context_size=self.CONFIG["context_size"],latent_size=self.CONFIG["latent_size"]).to(self.device)
        #self.belief_generator_sivae = BG.Generator(feature_size=self.CONFIG["feature_size"],context_size=self.CONFIG["context_size"],latent_size=self.CONFIG["latent_size"]).to(self.device)
        self.blender = Blender.Blender(context_size=self.CONFIG["context_size"], latent_size=self.CONFIG["latent_size"], score_min=self.CONFIG["dynamic_score_min"], score_max=self.CONFIG["dynamic_score_max"]).to(self.device)
        self.optimizer = optim.Adam([{'params':self.belief_generator.parameters(), 'lr':self.CONFIG["lr_generator"] * multi}, {'params':self.blender.parameters(), 'lr':self.CONFIG["lr_blender"]* multi}])
        #self.optimizer_sivae = optim.Adam([{'params':self.belief_generator_sivae.parameters(), 'lr':self.CONFIG["lr_generator"] * multi}, {'params':self.blender.parameters(), 'lr':self.CONFIG["lr_blender"]* multi}])

    def change_profile(self,random_generate = False, temperature = 1):
        self.profile = self.profile_generator.generate_synthetic_profile(random_generate, temperature)

    def _generate_belief(self, task_embeddings, profile_duplicate):
        recon_batch, mu, logvar = self.belief_generator(task_embeddings, profile_duplicate)
        #print('belief(mu, logvar):', mu, logvar)
        z_sample = self.belief_generator.reparameterize(mu, logvar)
        return z_sample

    def give_answer(self,reference_answer, task_embeddings, only_generative = False):
        obs_num = task_embeddings.shape[0]
        #print('profile:', self.profile)
        profile_duplicate = self.profile.unsqueeze(0).repeat(obs_num, 1)
        belief = self._generate_belief(task_embeddings, profile_duplicate)
        if only_generative:
            reference_answer = torch.full_like(reference_answer, self.CONFIG["fixed_ref_for_generator_only"]).to(
                self.device)
        #print('belief:', belief)
        #print(reference_answer.shape, belief.shape, profile_duplicate.shape)
        answer = self.blender(reference_answer, belief, profile_duplicate, self.CONFIG["diversity_param"])
        #print(answer.shape)

        return answer
