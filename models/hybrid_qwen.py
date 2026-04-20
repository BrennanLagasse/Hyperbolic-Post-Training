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

from transformers import PreTrainedModel, AutoModel
import torch.nn.functional as F

from geoopt.manifolds import Lorentz


class HyperbolicProjection(nn.Module):
    """ 
    Standard exponential map from the embedding space of dim d (tangent plane to hyperboloid)
    to d-dim hyperboloid embedded in d+1 dimensional space
    """

    def __init__(self, dim, k=1.0):
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

class HyperbolicQwen3(PreTrainedModel):
    config_class = HyperbolicQwenConfig

    def __init__(self, config):
        super().__init__(config)

        # Load in model without final prediction head (AutoModel vs AutoModelForCausalLM)
        self.backbone = AutoModel.from_pretrained(
            config.base_model_name
        )

        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size

        # Hyperbolic projection
        self.hyperbolic = HyperbolicProjection(
            dim=hidden_size,
            c=config.curvature
        )

        # Custom LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Optional: tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        if self.config.project_hidden:
            hidden_states = self.hyperbolic(hidden_states)

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