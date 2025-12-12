import torch
import torch.utils.data
from torch import nn

class Blender(nn.Module):
    def __init__(self, context_size, latent_size, score_min, score_max):  # context_size is dynamic
        super(Blender, self).__init__()
        self.latent_size = latent_size
        self.context_size = context_size
        self.score_min = score_min  # Store min score
        self.score_max = score_max  # Store max score

        self.fc = nn.Linear(latent_size + context_size, latent_size)  # Personal belief latent vector 'h'
        self.fcc = nn.Linear(latent_size, 10)  # Intermediate layer before bias/scale
        self.gc1 = nn.Linear(10, 1)  # Bias term
        # self.gc2 = nn.Linear(10, 1) # Original code had gc2 for diversity, but used fixed diversity later
        self.elu = nn.ELU()  # Not used in original forward, but common
        # self.sigmoid = nn.Sigmoid() # Not used

    def blend(self, yref, h_personal_belief, diversity_param):
        # h_personal_belief is the output of self.fc(inputs)
        hp = self.fcc(h_personal_belief)  # (bs, 10)
        bias = self.gc1(hp)  # (bs, 1)
        #torch.manual_seed(123)
        # Ensure yref is correctly shaped for broadcasting if bias is (bs,1) and yref is (bs,)
        if yref.ndim == 1:
            yref = yref.unsqueeze(1)  # Make it (bs, 1)

        eps = torch.randn_like(bias) * diversity_param
        score = yref + bias + eps  # Element-wise, yref might need unsqueeze
        return score.squeeze(1)  # Squeeze back to (bs,)

    def forward(self, yref, z, c, diversity_param):
        inputs = torch.cat([z, c], 1)
        h_personal_belief = self.fc(inputs)  # No activation in original code here, this is 'h'
        yscore_raw = self.blend(yref, h_personal_belief, diversity_param)
        yest = torch.clamp(yscore_raw, min=self.score_min, max=self.score_max)
        return yest
