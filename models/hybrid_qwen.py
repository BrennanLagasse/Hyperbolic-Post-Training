"""
Hybrid Qwen3-1.7B with hyperbolic projection layer

Replaces the standard linear (Euclidean) projection from hidden states to
vocabulary logits with a hyperbolic equivalent using the Lorentz (hyperboloid)
model.

Process (from Menglin):

  1. Hidden state x in R^d --> pad with zero time coordinate --> (0, x) \in R^{d+1}
  2. Map to hyperboloid via exponential map at the north pole
  3. Vocabulary embeddings (rows of the pretrained lm_head.weight) undergo
     the same procedure
  4. Logit for token i  =  −\langle exp_p(x), exp_p(w_i) \rangle_L

Initialization: spatial weights are copied directly from the pretrained
lm_head, so the model starts as a well-defined perturbation of the
Euclidean head (see the small-norm analysis in the derivation).
"""

import torch
import torch.nn as nn

from transformers import (
    PreTrainedModel, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoConfig,
)

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import torch.nn.functional as F

from geoopt.manifolds import Lorentz

def lorentz_inner(x, y):
    """ Compute lorentz inner product between x and y of shape [b, d+1] """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

class HyperbolicQwenConfig(Qwen3Config):
    model_type = "hyperbolic_qwen"

    def __init__(
        self,
        base_model_name="Qwen/Qwen3-1.7B",
        k=1.0,
        **kwargs
    ):
        """
        Args:
            base_model_name (str): name of base model on Huggingface
            k (float): negative curvature of the hyperbolic embedding space
        """
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.k = k

class HyperbolicProjection(nn.Module):
    """ 
    Standard exponential map from the embedding space of dim d (tangent plane to hyperboloid)
    to d-dim hyperboloid embedded in d+1 dimensional space
    """

    def __init__(self, k=1.0):
        """ 
        Params:
            dim (int) - dimension of the ambient space
            k (int) - *negative* curvature of the hyperboloid
        """
        super().__init__()

        assert k > 0, "Curvature of a hyperboloid is negative and k is the negative curvature, so it should be positive"

        self.hyp = Lorentz(k=k)

    def forward(self, x):
        """ 
        Compute exp_p(x) where p = (1/\sqrt{k}, 0, ..., 0)

        Params:
            x: tensor of shape [b, d] (in \R^n)
        
        Returns:
            x_hyp: tensor of shape [b, d+1] (in \H^n_k)
        """

        b, d = x.shape
        device = x.device
        dtype = x.dtype

        # Origin of the hyperboloid, p = (1/sqrt(k), 0, ..., 0)
        p = torch.zeros(size=(b, d+1), device=device, dtype=dtype)
        p[0] = 1 / (self.hyp.k ** 0.5)

        # Convert x to a derivative in ambient space at p by prepending a zero
        zeros = torch.zeros(size=(b, 1), device=device, dtype=dtype)
        x_amb = torch.cat((zeros, x), dim=1)

        # Apply exponential map
        x_hyp = self.hyp.expmap(p, x_amb)

        return x_hyp

class HyperbolicLMHead(nn.Module):
    """ 
    Implementation of hyperbolic LM head where weights and latent states
    are both stored in ambient space, projected to hyperboloid, and then
    logit scores are computed based on similarity
    """

    def __init__(self, config):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(config.vocab_size, config.hidden_size))
        self.hyp_proj = HyperbolicProjection(config.k)

    def forward(self, x):
        """
        Compute logit_k = <exp_p(x), exp_p(w_k)>_L
        """

        x_hyp = self.hyp_proj(x)
        weights_hyp = self.hyp_proj(self.weights)

        logits = lorentz_inner(x_hyp + weights_hyp)

        return logits

    def initialize_from_linear(self, linear_weights):
        with torch.no_grad():
            self.weights.copy_(linear_weights)
        
class HyperbolicQwen(PreTrainedModel):
    """ Qwen with hyperbolic projection head """

    config_class = HyperbolicQwenConfig
    all_tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)

        # Load in model without final prediction head (AutoModel vs AutoModelForCausalLM)
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        self.backbone = AutoModel.from_config(base_config)

        # LM Head (ambient embedding --> hyperbolic space --> alphabet)
        self.lm_head = HyperbolicLMHead(config)

        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):

        # Compute hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        # Project into hyperbolic space and eval
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

    # def initialize_from_pretrained(self):
    #     """ Load weights from base qwen model once """

    #     base_model = AutoModel.from_pretrained(self.config.base_model_name)
    #     self.backbone.load_state_dict(base_model.state_dict())

    #     base_lm = AutoModelForCausalLM.from_pretrained(self.config.base_model_name)
    #     original_head_weights = base_lm.lm_head.weight.data

    #     self.lm_head.initialize_from_linear(original_head_weights)

    def initialize_from_pretrained(self):
        """ Load weights from base qwen model once """

        print("New Initialization")

        base_lm = AutoModelForCausalLM.from_pretrained(self.config.base_model_name)

        # Load the base weights without the projection head
        self.backbone.load_state_dict(base_lm.model.state_dict())

        # Initialize projection head based on original projection head
        original_head_weights = base_lm.lm_head.weight.data
        self.lm_head.initialize_from_linear(original_head_weights)