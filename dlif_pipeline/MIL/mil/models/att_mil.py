import torch
from torch import nn
from typing import Optional, List, Union

from MIL.model.torch_utils import get_device


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    # Note: softmax not being applied here, as it will be applied later,
    # after masking out the padding.
    if n_latent == 0:
        return nn.Linear(n_in, 1)
    else:
        n_latent = n_latent or (n_in + 1) // 2
        return nn.Sequential(
            nn.Linear(n_in, n_latent),
            nn.Tanh(),
            nn.Linear(n_latent, 1)
        )


class MultiModal_Mixed_Attention_MIL(nn.Module):
    """Attention-based MIL model for mixed modality inputs with potential missing modalities."""

    use_lens = True

    def __init__(
        self,
        n_feats: List[int],
        n_out: int,
        z_dim: int = 256, # latent space dimension
        *,
        hidden_dim: Optional[Union[List[int], int]] = None, # same length as n_feats if None
        n_layers: Optional[Union[List[int], int]] = None, 
        dropout_p: float = 0.5,
        temperature: float = 1.
    ) -> None:
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = [z_dim] * len(n_feats)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * len(n_feats)
        if n_layers is None: # default is 1 layer
            n_layers = [1] * len(n_feats)
        if isinstance(n_layers, int):
            n_layers = [n_layers] * len(n_feats)

        # check if hidden_dim is same length as n_feats
        if len(hidden_dim) != len(n_feats) or len(n_layers) != len(n_feats):
            raise ValueError("hidden_dim and n_layers must be the same length as n_feats")
        
        self.n_modalities = len(n_feats)
        self.z_dim = z_dim
        self.n_feats = n_feats  # Save feature lengths for each modality
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create an encoder for each modality that projects to shared z_dim space
        self.encoders = nn.ModuleList()
        self.fc_mus = nn.ModuleList()
        self.fc_vars = nn.ModuleList()
        
        for i, nf in enumerate(n_feats):
            # Encoder layers
            layers = []
            # First layer (input to hidden)
            layers.extend([
                nn.Linear(in_features=nf, out_features=hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            
            # Middle layers (hidden to hidden)
            for _ in range(n_layers[i] - 1):
                layers.extend([
                    nn.Linear(in_features=hidden_dim[i], out_features=hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
            
            self.encoders.append(nn.Sequential(*layers))
            
            # Mean and variance projections
            self.fc_mus.append(nn.Linear(hidden_dim[i], z_dim))
            self.fc_vars.append(nn.Linear(hidden_dim[i], z_dim))
        
        # Create decoders that map the latent space into for each modality space
        self.decoders = nn.ModuleList()
        for i, nf in enumerate(n_feats):
            decoder_layers = []
            # First layer (latent to hidden)
            decoder_layers.extend([
                nn.Linear(in_features=z_dim, out_features=hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            
            # Middle layers (hidden to hidden)
            for _ in range(n_layers[i] - 1):
                decoder_layers.extend([
                    nn.Linear(in_features=hidden_dim[i], out_features=hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
            
            # Final projection back to original feature space
            decoder_layers.append(nn.Linear(in_features=hidden_dim[i], out_features=nf))
            
            self.decoders.append(nn.Sequential(*decoder_layers))

        # Single attention mechanism for the shared embedding space
        self.attention = Attention(z_dim)
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        
        self._neg_inf = torch.tensor(-torch.inf)
        self.temperature = temperature

    def forward(self, *inputs, attention=False, decode=True):
        """Forward pass through the network.
        
        Args:
            *inputs: Variable number of arguments where:
                - The first N-1 arguments are tensors for each modality
                - The last argument is the modality mask
        """
        modalities = inputs[:-1]  # All but last input are modality tensors
        modality_mask = inputs[-1]  # Last input is the mask
        
        # Encode each modality to means and variances
        mus, vars = self.encode_all(modalities)

        # Stack encoded vectors for attention
        # Shape: (batch_size, n_modalities, z_dim)
        stacked_embeddings = torch.stack(mus, dim=1)
        
        # Calculate masked attention scores
        masked_attention = self._masked_attention_scores(
            stacked_embeddings, 
            modality_mask,
            apply_softmax=True
        )
        
        # Apply attention weights to embeddings
        weighted_embeddings = (masked_attention * stacked_embeddings)
        
        # Sum across modalities
        pooled_embeddings = weighted_embeddings.sum(dim=1)  # Shape: (batch_size, z_dim)
        
        # Final prediction
        predictions = self.head(pooled_embeddings)

        if attention:
            res = predictions, masked_attention
        else:
            res = predictions

        if not self.training:
            return res

        if decode:
            # Get latent representations through reparameterization
            zs = self.reparameterize(mus, vars)
            # Decode latent representations back to modality space
            res = res, self.decode_all(zs), modalities, modality_mask

        return res

    def encode_all(self, modalities):
        """Encode all modalities to their means and variances in latent space.
        
        Args:
            modalities: List of tensors, one for each modality
            
        Returns:
            mus: List of mean tensors, one for each modality
            vars: List of variance tensors, one for each modality
        """
        mus, vars = [], []
        for i, modality in enumerate(modalities):
            h = self.encoders[i](modality)
            mu = self.fc_mus[i](h)
            var = self.fc_vars[i](h)
            mus.append(mu)
            vars.append(var)
        return mus, vars

    def decode_all(self, z):
        """Decode latent vectors back to all modality spaces.
        
        Args:
            z: Tensor of shape (batch_size, z_dim) or list of such tensors
            
        Returns:
            reconstructions: List of tensors, where reconstructions[i] contains 
                reconstructions into modality i's space.
                Shape is (batch_size, n_modalities, modality_i_dim)
        """
        if isinstance(z, list):
            reconstructions = [ 
                torch.stack([self.decoders[i](z_i) for z_i in z], dim=1) 
                for i in range(self.n_modalities)
            ] # (batch_size, n_modalities, modality_i_dim)

            return reconstructions

    def reparameterize(self, mus, logvars):
        """Reparameterize for each modality using the respective mu and logvar.
        
        Args:
            mus: List of mean tensors, one for each modality
            logvars: List of log variance tensors, one for each modality
            
        Returns:
            zs: List of sampled latent vectors, one for each modality
        """
        zs = []
        for mu, logvar in zip(mus, logvars):
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(mean=0, std=1, size=std.size(), device=mu.device)
            z = eps.mul(std).add_(mu)
            zs.append(z)
        return zs

    def _masked_attention_scores(self, embeddings, modality_mask, *, apply_softmax=True):
        """Calculate masked attention scores.
        
        Args:
            embeddings: Tensor of shape (batch_size, n_modalities, z_dim)
            modality_mask: Tensor of shape (batch_size, n_modalities)
            apply_softmax: Whether to apply softmax to the masked attention scores
        
        Returns:
            Tensor of shape (batch_size, n_modalities, 1) containing attention scores
        """
        attention_scores = self.attention(embeddings)  # Shape: (batch_size, n_modalities, 1)
        
        # Mask out attention for missing modalities
        masked_attention = torch.where(
            modality_mask.unsqueeze(-1),
            attention_scores,
            torch.full_like(attention_scores, self._neg_inf)
        )
        
        if apply_softmax:
            return torch.softmax(masked_attention / self.temperature, dim=1)
        return masked_attention

    def relocate(self):
        device = get_device()
        self.to(device)
        self._neg_inf = self._neg_inf.to(device)

    def plot(*args, **kwargs):
        pass

