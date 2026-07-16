#!/usr/bin/env python3
"""First test script""" 

# Import libraries
from pathlib import Path

import lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Optional

# Import models and data-loadings
from src.data.event_jets_data_loading import EventJetsL1TriggerDataset
from src.data.event_particles_data_loading import EventPartL1TriggerDataset
from src.data.jet_constituents_data_loading import JetConstL1TriggerDataset
from src.models.mlp_vqvae import MLPVQVAE
from src.models.transformer_vqvae import TransformerVQVAE


# Inverse preprocessing function
def inverse_preprocessing(
    feats: torch.Tensor,
    mask: torch.Tensor,
    jet_feats: Optional[torch.Tensor] = None,
):
    """
    Inverse preprocessing for constituent features.

    Args:
        feats:
            Tensor of shape [B, N, 3]
            containing:
                [:,:,0] -> preprocessed pt
                [:,:,1] -> preprocessed eta
                [:,:,2] -> preprocessed phi
        
        mask:
            Tensor of shape [B, N]
            containing True entris for valid constituents

        jet_feats:
            Tensor of shape [B, 4]
            containing jet features.

    Returns:
        Tensor of shape [B, N, 3]
        with original:
            pt, eta, phi
    """
    
    # Clone the features tensor
    out = feats.clone()
    
    # Unsqueeze mask
    mask3d = mask.unsqueeze(-1)  # [B, N, 1]

    # PT inverse
    pt = torch.exp(out[:, :, 0] + 1.8) - 1e-8

    # Eta inverse
    eta = out[:, :, 1] * 3.0
    
    if jet_feats is None:
        # Phi inverse
        phi_sin = out[:, :, 2]
        phi_cos = out[:, :, 3]

        phi = torch.atan2(phi_sin,phi_cos)
    else:
        phi = out[:, :, 2] * 3.0
    
    # If Jet-Constituents-Level, recover absolute eta and phi
    if jet_feats is not None:
        
        jet_eta = jet_feats[:, 1]
        jet_phi = jet_feats[:, 2]
        
        eta += jet_eta[:, None]
        phi += jet_phi[:, None]

        # wrap phi to [-pi, pi]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

    # Stack
    recovered = torch.stack([pt, eta, phi], dim=-1)
    
    # Keep padding entries to zero
    recovered = torch.where(
        mask3d,
        recovered,
        torch.zeros_like(recovered),
    )
    
    # Return recovered features
    return recovered


def main():

    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)


    ########################################
    #                                      #
    #           INPUT VARIABLES            #
    #                                      #
    ########################################
    
    # Data loader selection-----------------

    print("1) Jet-Constituents-level")
    print("2) Event-Particles-level")
    print("3) Event-Jets-level")
    
    dl = input("Enter the data loader number (1, 2, or 3): ")

    while dl not in ["1", "2", "3"]:
        dl = input("INVALID ENTRY! Enter the data loader number (1, 2, or 3): ")

    if dl == "1":
        path_data = "jet_const"
    elif dl == "2":
        path_data = "event_part"
    elif dl == "3":
        path_data = "event_jets"

    # Model selection-----------------------
    
    print("1) MLP-VQVAE")
    print("2) Transformer-VQVAE")

    mod = input("Enter the model number (1 or 2): ")

    while mod not in ["1", "2"]:
        mod = input("INVALID ENTRY! Enter the model number (1 or 2): ")

    if mod == "1":
        path_model = "mlp"
    elif mod == "2":
        path_model = "transformer"
    
    # Rotation Trick for checkpoint's path--

    print("1) False")
    print("2) True")

    r = input("Enter the Rotation Trick number (1 or 2): ")

    while r not in ["1", "2"]:
        r = input("INVALID ENTRY! Enter the Rotation Trick number (1 or 2): ")

    if r == "1":
        path_rot = "False"
    elif r == "2":
        path_rot = "True"

    # CB size for checkpoint's path---------
    
    print("1) 256")
    print("2) 512")
    print("3) 1024")

    cb = input("Enter the codebook size number (1, 2 or 3): ")

    while cb not in ["1", "2", "3"]:
        cb = input("INVALID ENTRY! Enter the model number (1, 2 or 3): ")

    if cb == "1":
        path_cb = "256"
    elif cb == "2":
        path_cb = "512"
    elif cb == "3":
        path_cb = "1024"

    # Checkpoint selection------------------

    checkpoint = input("Enter the checkpoint's name: ")
    checkpoint = f"checkpoints/{path_model}-{path_data}-{path_rot}-{path_cb}/" + checkpoint

    path = Path(checkpoint)

    
    ########################################
    #                                      #
    #        DATA-LOADING SELECTION        #
    #                                      #
    ########################################

    # JET-CONSTITUENTS-LEVEL----------------
    
    if dl == "1":
        # Dataset not preprocessed 
        dataset_not_prep = JetConstL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_particles=128,
            features=["L1T_PUPPIPart_PT",
                    "L1T_PUPPIPart_Eta",
                    "L1T_PUPPIPart_Phi",
                    "L1T_PUPPIPart_PuppiW",
                    "L1T_JetPuppiAK4_PT",
                    "L1T_JetPuppiAK4_Eta",
                    "L1T_JetPuppiAK4_Phi",
                    "L1T_JetPuppiAK4_Mass",
                    "L1T_JetPuppiAK4_ConstituentsIdx"
            ],
            preprocessing=False
        )

        # Dataset preprocessed
        dataset_prep = JetConstL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_particles=128,
            features=["L1T_PUPPIPart_PT",
                    "L1T_PUPPIPart_Eta",
                    "L1T_PUPPIPart_Phi",
                    "L1T_PUPPIPart_PuppiW",
                    "L1T_JetPuppiAK4_PT",
                    "L1T_JetPuppiAK4_Eta",
                    "L1T_JetPuppiAK4_Phi",
                    "L1T_JetPuppiAK4_Mass",
                    "L1T_JetPuppiAK4_ConstituentsIdx"
            ],
            preprocessing=True
        )

        # Dataloader not preprocessed
        data_loader_not_prep = torch.utils.data.DataLoader(
            dataset_not_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )

        # Dataloader preprocessed
        data_loader_prep = torch.utils.data.DataLoader(
            dataset_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )
    
    # EVENT-PARTICLES-LEVEL-----------------

    elif dl == "2":
    
        # Dataset not preprocessed 
        dataset_not_prep = EventPartL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_particles=128,
            features=["L1T_PUPPIPart_PT",
                    "L1T_PUPPIPart_Eta",
                    "L1T_PUPPIPart_Phi",
                    "L1T_PUPPIPart_PuppiW",
            ],
            puppiw_threshold=0.05,
            preprocessing=False
        )

        # Dataset preprocessed 
        dataset_prep = EventPartL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_particles=128,
            features=["L1T_PUPPIPart_PT",
                    "L1T_PUPPIPart_Eta",
                    "L1T_PUPPIPart_Phi",
                    "L1T_PUPPIPart_PuppiW",
            ],
            puppiw_threshold=0.05,
            preprocessing=True
        )

        # Dataloader not preprocessed
        data_loader_not_prep = torch.utils.data.DataLoader(
            dataset_not_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )

        # Dataloader preprocessed
        data_loader_prep = torch.utils.data.DataLoader(
            dataset_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )

    # EVENT-JETS-LEVEL----------------------

    elif dl == "3":
    
        # Dataset not preprocessed 
        dataset_not_prep = EventJetsL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_jets=16,
            features=["L1T_JetPuppiAK4_PT",
                    "L1T_JetPuppiAK4_Eta",
                    "L1T_JetPuppiAK4_Phi",
            ],
            preprocessing=False
        )

        # Dataset preprocessed 
        dataset_prep = EventJetsL1TriggerDataset(
            parquet_dirs="/run/media/francesco/STORAGE/data_cern_2/Test",
            max_jets=16,
            features=["L1T_JetPuppiAK4_PT",
                    "L1T_JetPuppiAK4_Eta",
                    "L1T_JetPuppiAK4_Phi",
            ],
            preprocessing=True
        )

        # Dataloader not preprocessed
        data_loader_not_prep = torch.utils.data.DataLoader(
            dataset_not_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )

        # Dataloader preprocessed
        data_loader_prep = torch.utils.data.DataLoader(
            dataset_prep,
            batch_size=32,
            num_workers=0,
            pin_memory=True
        )
    

    ########################################
    #                                      #
    #            MODEL SELECTION           #
    #                                      #
    ########################################    

    # MLP-VQVAE-----------------------------

    if mod == "1":
        model = MLPVQVAE.load_from_checkpoint(path, weights_only=False)

    # Transformer-VQVAE---------------------

    elif mod == "2":
        model = TransformerVQVAE.load_from_checkpoint(path, weights_only=False)
    

    ########################################
    #                                      #
    #        LISTS INITIALIZATION          #
    #                                      #
    ########################################
    
    # Will contain original features
    pt_orig = []
    eta_orig = []
    phi_orig = []
    
    # Will contain reconstructed features
    pt_reco = []
    eta_reco = []
    phi_reco = []
    
    # Will contain codebook indices
    idx = []
    

    ########################################
    #                                      #
    #           MODEL RUNNING              #
    #                                      #
    ########################################

    # Set model in evaluation mode
    model.eval()
    
    # Select GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Collect original features
    for batch in tqdm(data_loader_not_prep, desc="Looking for original features"):
        
        if dl == "1":
            x, m, j = batch

        else:
            x, m = batch

        pt_o = x[:, :, 0]
        eta_o = x[:, :, 1]
        phi_o = x[:, :, 2]

        pt_orig.extend(pt_o[m].cpu())
        eta_orig.extend(eta_o[m].cpu())
        phi_orig.extend(phi_o[m].cpu())
    
    # Collect reconstructed features and codebook indices
    for batch in tqdm(data_loader_prep, desc="Computing reconstructed features"):
        
        if dl == "1":
            x, m, j = batch

            x = x.to(device)
            m = m.to(device)
            j = j.to(device)
        
        else:
            x, m = batch

            x = x.to(device)
            m = m.to(device)        

        with torch.no_grad():

            output = model(x,m)
            
            if dl == "1":
                reco = inverse_preprocessing(output[0], m, j)
            
            else:
                reco = inverse_preprocessing(output[0], m)
            
            pt_r = reco[:, :, 0]
            eta_r = reco[:, :, 1]
            phi_r = reco[:, :, 2]

            pt_reco.extend(pt_r[m].cpu())
            eta_reco.extend(eta_r[m].cpu())
            phi_reco.extend(phi_r[m].cpu())

            #idx.extend(output[2].flatten().cpu())
            idx.extend(output[2][m].cpu())


    ########################################
    #                                      #
    #             PLOTTING                 #
    #                                      #
    ########################################

    print("Plotting...")

    # Hyperparameters
    ckpt = torch.load(path, weights_only=False)
    model_name = str(ckpt["hyper_parameters"]["cfg"]["name"])
    cb_size = str(ckpt["hyper_parameters"]["cfg"]["codebook_size"])
    rot = str(ckpt["hyper_parameters"]["cfg"]["rotation_trick"])
    
    # Codebook usage
    cb_usage = len(torch.unique(torch.stack(idx))) / int(cb_size)
    
    # pT bins
    bins = np.histogram_bin_edges(np.concatenate([pt_orig, pt_reco]), bins=50)
    
    # Initialize the multiplots figure
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # PT plot on log scale
    ax[0,0].hist(pt_orig, density=True, bins=bins, color="orange", label="Original", log=True, alpha=0.7)
    ax[0,0].hist(pt_reco, density=True, bins=bins, histtype="step", color="purple", label="Reconstructed", log=True)
    ax[0,0].set_xlabel(r"$p_T$ [GeV]")
    ax[0,0].set_ylabel("Density")
    ax[0,0].set_title(r"Distribution of $p_T$ [GeV] on log scale")
    ax[0,0].legend()
    
    # Eta plot
    ax[0,1].hist(eta_orig, density=True, bins=50, color="red", label="Original", alpha=0.7)
    ax[0,1].hist(eta_reco, density=True, bins=50, histtype="step", color="purple", label="Reconstructed")
    ax[0,1].set_xlabel(r"$\eta$")
    ax[0,1].set_ylabel("Density")
    ax[0,1].set_title(r"Distribution of $\eta$")
    ax[0,1].legend()
    
    # Phi plot
    ax[0,2].hist(phi_orig, density=True, bins=50, color="blue", label="Original", alpha=0.7)
    ax[0,2].hist(phi_reco, density=True, bins=50, histtype="step", color="purple", label="Reconstructed")
    ax[0,2].set_xlabel(r"$\phi$")
    ax[0,2].set_ylabel("Density")
    ax[0,2].set_title(r"Distribution of $\phi$")
    ax[0,2].legend()

    # PT plot no log scale
    ax[1,0].hist(pt_orig, density=True, bins=bins, color="orange", label="Original", alpha=0.7)
    ax[1,0].hist(pt_reco, density=True, bins=bins, histtype="step", color="purple", label="Reconstructed")
    ax[1,0].set_xlabel(r"$p_T$ [GeV]")
    ax[1,0].set_ylabel("Density")
    ax[1,0].set_title(r"Distribution of $p_T$ [GeV]")
    ax[1,0].legend()

    # PT plot (cut on x axis)
    p99 = np.percentile(np.array(pt_orig), 99) + 10
    min = np.min(np.array(pt_orig)) - 5

    ax[1,2].hist(pt_orig, density=True, bins=bins, color="orange", label="Original", alpha=0.7)
    ax[1,2].hist(pt_reco, density=True, bins=bins, histtype="step", color="purple", label="Reconstructed")
    ax[1,2].set_xlabel(r"$p_T$ [GeV]")
    ax[1,2].set_ylabel("Density")
    ax[1,2].set_xlim(min, p99)
    ax[1,2].set_title(r"Distribution of $p_T$ [GeV] (cut on x axis)")
    ax[1,2].legend()
    
    # Codebook usage plot
    ax[1,1].hist(idx, density=True, bins=int(cb_size), color="brown", alpha=0.9)
    ax[1,1].set_xlim(0, int(cb_size))
    ax[1,1].set_xlabel(f"Quantization index (CB-usage={cb_usage})")
    ax[1,1].set_ylabel("Density")
    ax[1,1].set_title("Distribution of the quantization indices")

    #ax[1, 2].axis('off')
    
    plt.suptitle(model_name + " VQ-VAE, CB_size: " + cb_size + ", Rotation_trick: " + rot)
    plt.tight_layout()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()