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

def lorentz_inner(x, w):
    """
    Compute Lorentz inner product <x, w>_L = -x0*w0 + x1*w1 + ... + xd*wd
    
    Args:
        x: [..., d+1] points on hyperboloid (batch)
        w: [vocab_size, d+1] projected weight vectors
    
    Returns:
        logits: [..., vocab_size]
    """
    # Ensure w is on same device and dtype as x
    w = w.to(device=x.device, dtype=x.dtype)
    
    # Split time and spatial components
    x_time = x[..., :1]        # [..., 1]
    x_space = x[..., 1:]       # [..., d]
    
    w_time = w[:, :1]          # [vocab, 1]
    w_space = w[:, 1:]         # [vocab, d]
    
    # Lorentz inner: -<time components> + <spatial components>
    time_term = x_time @ w_time.T      # [..., vocab]
    space_term = x_space @ w_space.T   # [..., vocab]
    
    return -time_term + space_term

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

        self.k = k
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
        p[..., 0] = 1 / (self.k ** 0.5)

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
        
        # Cache for projected weights
        self._weight_hyp_cache = None
        self._weight_hash = None

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

        weight_hyp = self._get_weight_hyp()

        logits = lorentz_inner(x_hyp, weight_hyp)

        return logits

    def _get_weight_hyp(self):
        """Return cached projected weights, recomputing only if weights have changed."""

        # During training recompute projection of weights every step since they are updated
        if self.training:
            return self.hyp_proj(self.weight)
        
        # During inference, cache 
        current_hash = self.weight.data_ptr()
        if self._weight_hyp_cache is None or current_hash != self._weight_hash:
            self._weight_hyp_cache = self.hyp_proj(self.weight)
            self._weight_hash = current_hash
        
        return self._weight_hyp_cache

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