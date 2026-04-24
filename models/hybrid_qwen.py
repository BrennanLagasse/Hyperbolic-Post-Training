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
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

import torch.nn.functional as F

from geoopt.manifolds import Lorentz

from transformers.modeling_outputs import CausalLMOutputWithPast

import math

# def lorentz_inner(x, y):
#     """ Compute lorentz inner product between x and y of shape [b, d+1] """
#     return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

def lorentz_inner(x, w):
    """
    x: [b, l, d+1]
    w: [v, d+1]

    returns: [b, l, v]
    """

    # Just reverse sign of one time component then compute as matmul

    x_sign = x.clone()
    x_sign[..., 0] = -x_sign[..., 0]

    return x_sign @ w.T

class HyperbolicQwenConfig(Qwen3Config):
    model_type = "hybrid_qwen"

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
        self.tie_word_embeddings=False

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
            x: tensor of shape [..., d] (in \R^n)
        
        Returns:
            x_hyp: tensor of shape [..., d+1] (in \H^n_k)
        """

        *batch_dims, d = x.shape
        device = x.device
        dtype = x.dtype

        # Origin of the hyperboloid, p = (1/sqrt(k), 0, ..., 0)
        p = torch.zeros(size=(*batch_dims, d+1), device=device, dtype=dtype)
        p[..., 0] = 1 / (self.hyp.k ** 0.5)

        # Convert x to a derivative in ambient space at p by prepending a zero
        zeros = torch.zeros(size=(*batch_dims, 1), device=device, dtype=dtype)
        x_amb = torch.cat((zeros, x), dim=-1)

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

        self.weight = nn.Parameter(torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16))
        self.hyp_proj = HyperbolicProjection(config.k)

    def forward(self, x):
        """
        Compute logit_k = <exp_p(x), exp_p(w_k)>_L

        where x_final is the embedding of the last token in x

        Params:
            x: [batch, len, emb_dim] or [batch, 1, emb_dim]

        Returns:
            logits: [batch, alphabet_size]
        """

        # Normalize to avoid exponential 
        x = x / math.sqrt(x.shape[-1])

        x_hyp = self.hyp_proj(x)

        weight_hyp = self.hyp_proj(self.weight)

        logits = lorentz_inner(x_hyp, weight_hyp)

        # print(f"x: {x}")
        # print(f"w: {self.weight}")

        # print(f"fwd: {logits}")

        # print(f"TX: {x.norm(dim=-1, keepdim=False)}")
        # print(f"TW: {self.weight.norm(dim=-1, keepdim=False)}")
        # print(f"HX: {x_hyp.norm(dim=-1, keepdim=False)}")
        # print(f"HW: {weight_hyp.norm(dim=-1, keepdim=False)}")

        return logits

    # def save_pretrained(self, path):
    #     self.model.save_pretrained(path)

    # @classmethod
    # def from_pretrained(cls, model_name: str, compute_dtype=torch.bfloat16):
    #     instance = cls(model_name, compute_dtype)
    #     return instance

    def initialize_from_linear(self, linear_weight):
        with torch.no_grad():
            self.weight.copy_(linear_weight)
        
class HyperbolicQwen(Qwen3ForCausalLM):
    """ Qwen with hyperbolic projection head """

    config_class = HyperbolicQwenConfig
    all_tied_weights_keys = []
    tie_word_embeddings = False

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
        past_key_values=None,
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

        # print(f"labels: {labels}")
        # print(f"logits: {logits}")
        # print(f"loss: {loss}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def initialize_from_pretrained(self):
        """ Load weights from base qwen model once """

        print("New Initialization")

        base_lm = AutoModelForCausalLM.from_pretrained(self.config.base_model_name)

        # Load the base weights without the projection head
        self.backbone.load_state_dict(base_lm.model.state_dict())

        # Initialize projection head based on original projection head
        original_head_weights = base_lm.lm_head.weight.data
        self.lm_head.initialize_from_linear(original_head_weights)

        print(self.lm_head.weight)

# Initialize the models
print("Registering HyperbolicQwen")

AutoConfig.register(
    "hybrid_qwen", 
    HyperbolicQwenConfig
)

AutoModelForCausalLM.register(
    HyperbolicQwenConfig, 
    HyperbolicQwen
)