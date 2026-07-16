#!/usr/bin/env python3
"""Function that inverses the preprocessing of the features"""

import torch


def inverse_preprocess(feats, mask, jet_feats=None):
    """Inverse: converts normalized features back to physical units (pt, eta, phi)"""
    feats = feats.clone()

    # Inverse pT: exp(x + 1.8)
    feats[:,:,0] = torch.exp(feats[:,:,0] + 1.8) - 1e-8
    # Inverse eta: x * 3
    feats[:,:,1] = feats[:,:,1] * 3.0

    # Inverse phi: depende se jet_feats è None (sin/cos) o no (scaled)
    if jet_feats is None:
        phi = torch.atan2(feats[:,:,2], feats[:,:,3])
        feats[:,:,2] = phi
        feats = feats[:,:,:3]  # Remove cos(phi) channel
    else:
        feats[:,:,2] = feats[:,:,2] * 3.0
        # Add back jet relative offsets here if needed
        jet_eta = jet_feats[:, 1]
        jet_phi = jet_feats[:, 2]
        
        feats[:,:,1] += jet_eta[:, None]
        feats[:,:,2] += jet_phi[:, None]

        # wrap phi to [-pi, pi]
        feats[:,:,2] = (feats[:,:,2] + torch.pi) % (2 * torch.pi) - torch.pi

    return feats * mask.unsqueeze(-1)