import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as pl
from typing import Optional

from src.data.inverse_preprocessing import inverse_preprocess

class HistogramPlotter(pl.Callback):
      
    """Callback to plot and log histograms of reconstructed vs original features at end of each validation epoch."""

    def __init__(
        self,
        data_loading: str,
        cb_size: int,
        model_name: str,
        rotation: str,
        output_dir: str = "validation_plots",
        log_every_n_epochs: int = 1,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_loading: String indicating the type of data loading used
            model_name: Name of the model being trained
            rotation: String indicating if thr rotation trick is used
            output_dir: Directory to save plots
            log_every_n_epochs: How often to generate plots
            max_samples: Maximum number of samples to use for plotting (for memory efficiency)
        """
        self.data_loading = data_loading
        self.model_name = model_name
        self.rotation = rotation
        self.output_dir = output_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.originals = []
        self.reconstructions = []
        self.masks = []
        self.jet_feats = []
        self.idx = []
        self.cb_size = cb_size

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect original and reconstructed features at each validation batch."""
        if self.data_loading == "jet_const":
            x, mask, j = batch
            self.jet_feats.append(j.detach().clone())
        else:
            x, mask = batch  
        
        outputs, idx = outputs

        self.originals.append(x.detach().clone())
        self.masks.append(mask.detach().clone())
        self.reconstructions.append(outputs.detach().clone()) 
        self.idx.append(idx.detach().clone())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Plot histograms at end of validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Concatenate all batches
        original = torch.cat(self.originals, dim=0)
        reconstruction = torch.cat(self.reconstructions, dim=0)
        mask = torch.cat(self.masks, dim=0)
        idx = torch.cat(self.idx, dim=0)

        if self.data_loading == "jet_const":
            jet_feats = torch.cat(self.jet_feats, dim=0)

        # Optionally limit samples
        if self.max_samples is not None:
            n = min(len(original), self.max_samples)
            original = original[:n]
            reconstruction = reconstruction[:n]
            mask = mask[:n]
            idx = idx[:n]
            jet_feats = jet_feats[:n] if self.data_loading == "jet_const" else None
        
        original_post = inverse_preprocess(original, mask, True if self.data_loading == "event_jets" else False, jet_feats if self.data_loading == "jet_const" else None)
        reconstruction_post = inverse_preprocess(reconstruction, mask, True if self.data_loading == "event_jets" else False, jet_feats if self.data_loading == "jet_const" else None)
        
        # Apply mask and flatten
        mask3d = mask.unsqueeze(-1)
        orig_flat = original[mask].cpu()
        orig_post_flat = original_post[mask].cpu()
        recon_flat = reconstruction[mask].cpu()
        recon_post_flat = reconstruction_post[mask].cpu()
        idx = idx[mask].cpu()

        # Create plots
        fig_pre = self._create_histograms(orig_flat, recon_flat, trainer.current_epoch, post=False)
        fig_post = self._create_histograms(orig_post_flat, recon_post_flat, trainer.current_epoch, post=True)
        fig_cb = self._create_cb_usage(idx, trainer.current_epoch)

        # Log to TensorBoard
        if hasattr(pl_module, 'logger') and pl_module.logger:
            pl_module.logger.experiment.add_figure(
                'validation/histograms_pre', fig_pre, trainer.current_epoch
            )

        if hasattr(pl_module, 'logger') and pl_module.logger:
            pl_module.logger.experiment.add_figure(
                'validation/histograms_post', fig_post, trainer.current_epoch
            )

        if hasattr(pl_module, 'logger') and pl_module.logger:
            pl_module.logger.experiment.add_figure(
                'validation/cb_plot', fig_cb, trainer.current_epoch
            )

        # Also save to disk
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'epoch_{trainer.current_epoch:02d}_pre.png')
        fig_pre.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_pre)

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'epoch_{trainer.current_epoch:02d}_post.png')
        fig_post.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_post)

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'epoch_{trainer.current_epoch:02d}_cb.png')
        fig_cb.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_cb)

        # Clear stored data
        self.originals.clear()
        self.reconstructions.clear()
        self.masks.clear()
        self.jet_feats.clear()
        self.idx.clear()

    def _create_histograms(self, original, reconstruction, epoch, post):
        """Create histogram comparison plots."""
        n_features = original.shape[-1]

        if n_features == 4 and (self.data_loading == "event_jets" or self.data_loading == "event_part"):
            feature_names = [r'$P_{t}$', r'$\eta$', r'$\cos(\phi)$', r'$\sin(\phi)$']
            color = ['orange', 'red', 'blue', 'blue']

        else:
            feature_names = [r'$P_{t}$', r'$\eta$', r'$\phi$']
            color = ['orange', 'red', 'blue']

        # Create subplots: 2 rows x n_features columns
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 6))
        if n_features == 1:
            axes = axes.reshape(1, -1)

        for i, (feat_name, ax_pre) in enumerate(zip(feature_names, axes)):
            # Get feature data
            orig_feat = original[:, i]
            reco_feat = reconstruction[:, i]

            # Find bins
            all_data = torch.cat([orig_feat, reco_feat]).numpy()
            #all_data = all_data[all_data != 0]  # Remove zeros (padding)
            if len(all_data) == 0:
                continue

            bins = np.histogram_bin_edges(all_data, bins=50)

            # Original vs reconstructed preprocessed (top row)
            ax_pre.hist(orig_feat.numpy(), bins=bins, density=True, log=True if (post and i == 0) or (not post and i in (1,2) and self.data_loading == "jet_const") else False,
                        color=color[i], label='Original', alpha=0.7)
            ax_pre.hist(reco_feat.numpy(), bins=bins, density=True, log=True if (post and i == 0) or (not post and i in (1,2) and self.data_loading == "jet_const") else False,
                        color='purple', label='Reconstructed', alpha=0.9, histtype='step')
            #ax_pre.set_title(f'{feat_name}')
            ax_pre.set_xlabel(feat_name)
            ax_pre.set_ylabel('Density')
            ax_pre.legend()

        fig.suptitle(f'Validation Epoch {epoch} - {self.model_name}-VQVAE-rot:{self.rotation}-cb:{self.cb_size}')
        fig.tight_layout()

        return fig
    
    def _create_cb_usage(self, idx, epoch):
        """Create codebook usage plot."""
        
        # Codebook usage
        cb_usage = len(torch.unique(idx)) / int(self.cb_size)
        
        fig, ax = plt.subplots(figsize=(8,5))

        bins = np.arange(self.cb_size + 1) - 0.5
        
        # Codebook usage plot
        ax.hist(idx.numpy(), density=True, bins=bins, color="brown", alpha=0.8)
        ax.set_xlim(-0.5, self.cb_size - 0.5)
        ax.set_xlabel(f"Quantization index (CB-usage={cb_usage})")
        ax.set_ylabel("Density")        

        fig.suptitle(f'Validation Epoch {epoch} - {self.model_name}-VQVAE-rot:{self.rotation}-cb:{self.cb_size}')
        fig.tight_layout()

        return fig
    


# HistogramTestPlotter
class HistogramTestPlotter(pl.Callback):
    """
    Callback to plot and log histograms of reconstructed vs original
    features at the end of the test epoch.
    """

    def __init__(
        self,
        data_loading: str,
        cb_size: int,
        model_name: str,
        rotation: str,
        output_dir: str = "test_plots",
        max_samples=None,
    ):
        """
        Args:
            data_loading: String indicating the type of data loading used
            cb_size: Codebook size
            model_name: Name of the model
            rotation: String indicating if the rotation trick is used
            output_dir: Directory where plots are saved
            max_samples: Maximum number of samples to use for plotting
        """

        self.data_loading = data_loading
        self.model_name = model_name
        self.rotation = rotation
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.cb_size = cb_size

        self.originals = []
        self.reconstructions = []
        self.masks = []
        self.jet_feats = []
        self.idx = []

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx
    ):
        """
        Collect original and reconstructed features
        from each test batch.
        """

        if self.data_loading == "jet_const":
            x, mask, j = batch
            self.jet_feats.append(j.detach().clone())
        else:
            x, mask = batch

        # Expected:
        # outputs = (reconstruction, quantization_indices)
        reconstruction, idx = outputs

        self.originals.append(
            x.detach().clone()
        )

        self.masks.append(
            mask.detach().clone()
        )

        self.reconstructions.append(
            reconstruction.detach().clone()
        )

        self.idx.append(
            idx.detach().clone()
        )

    def on_test_epoch_end(self, trainer, pl_module):
        """
        Create histograms at the end of the test epoch.
        """

        # -----------------------------------------------------
        # Concatenate all batches
        # -----------------------------------------------------

        original = torch.cat(
            self.originals,
            dim=0
        )

        reconstruction = torch.cat(
            self.reconstructions,
            dim=0
        )

        mask = torch.cat(
            self.masks,
            dim=0
        )

        idx = torch.cat(
            self.idx,
            dim=0
        )

        if self.data_loading == "jet_const":
            jet_feats = torch.cat(
                self.jet_feats,
                dim=0
            )
        else:
            jet_feats = None

        # -----------------------------------------------------
        # Limit number of samples
        # -----------------------------------------------------

        if self.max_samples is not None:

            n = min(
                len(original),
                self.max_samples
            )

            original = original[:n]
            reconstruction = reconstruction[:n]
            mask = mask[:n]
            idx = idx[:n]

            if jet_feats is not None:
                jet_feats = jet_feats[:n]

        # -----------------------------------------------------
        # Inverse preprocessing
        # -----------------------------------------------------

        original_post = inverse_preprocess(
            original,
            mask,
            True if self.data_loading == "event_jets" else False,
            jet_feats if self.data_loading == "jet_const" else None
        )

        reconstruction_post = inverse_preprocess(
            reconstruction,
            mask,
            True if self.data_loading == "event_jets" else False,
            jet_feats if self.data_loading == "jet_const" else None
        )

        # -----------------------------------------------------
        # Apply mask and flatten
        # -----------------------------------------------------

        orig_flat = original[mask].cpu()
        orig_post_flat = original_post[mask].cpu()

        recon_flat = reconstruction[mask].cpu()
        recon_post_flat = reconstruction_post[mask].cpu()

        idx_flat = idx[mask].cpu()

        # -----------------------------------------------------
        # Create plots
        # -----------------------------------------------------

        fig_pre = self._create_histograms(
            orig_flat,
            recon_flat,
            post=False
        )

        fig_post = self._create_histograms(
            orig_post_flat,
            recon_post_flat,
            post=True
        )

        fig_cb = self._create_cb_usage(
            idx_flat
        )

        # -----------------------------------------------------
        # Log to TensorBoard
        # -----------------------------------------------------

        if hasattr(pl_module, "logger") and pl_module.logger:

            pl_module.logger.experiment.add_figure(
                "test/histograms_pre",
                fig_pre,
                trainer.global_step
            )

            pl_module.logger.experiment.add_figure(
                "test/histograms_post",
                fig_post,
                trainer.global_step
            )

            pl_module.logger.experiment.add_figure(
                "test/cb_plot",
                fig_cb,
                trainer.global_step
            )

        # -----------------------------------------------------
        # Save plots to disk
        # -----------------------------------------------------

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        filepath = os.path.join(
            self.output_dir,
            "test_pre.png"
        )

        fig_pre.savefig(
            filepath,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig_pre)

        filepath = os.path.join(
            self.output_dir,
            "test_post.png"
        )

        fig_post.savefig(
            filepath,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig_post)

        filepath = os.path.join(
            self.output_dir,
            "test_cb.png"
        )

        fig_cb.savefig(
            filepath,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig_cb)

        # -----------------------------------------------------
        # Clear stored data
        # -----------------------------------------------------

        self.originals.clear()
        self.reconstructions.clear()
        self.masks.clear()
        self.jet_feats.clear()
        self.idx.clear()

    # ---------------------------------------------------------
    # HISTOGRAMS
    # ---------------------------------------------------------

    def _create_histograms(
        self,
        original,
        reconstruction,
        post
    ):
        """Create histogram comparison plots."""

        n_features = original.shape[-1]

        if (
            n_features == 4
            and self.data_loading in (
                "event_jets",
                "event_part"
            )
        ):
            feature_names = [
                r"$P_t$",
                r"$\eta$",
                r"$\cos(\phi)$",
                r"$\sin(\phi)$"
            ]

            color = [
                "orange",
                "red",
                "blue",
                "blue"
            ]

        else:
            feature_names = [
                r"$P_t$",
                r"$\eta$",
                r"$\phi$"
            ]

            color = [
                "orange",
                "red",
                "blue"
            ]

        # -----------------------------------------------------
        # Create figure
        # -----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            n_features,
            figsize=(5 * n_features, 6)
        )

        if n_features == 1:
            axes = [axes]

        # -----------------------------------------------------
        # Plot each feature
        # -----------------------------------------------------

        for i, (feat_name, ax) in enumerate(
            zip(feature_names, axes)
        ):

            orig_feat = original[:, i]
            reco_feat = reconstruction[:, i]

            all_data = torch.cat(
                [orig_feat, reco_feat]
            ).numpy()

            if len(all_data) == 0:
                continue

            bins = np.histogram_bin_edges(
                all_data,
                bins=50
            )

            log_scale = (
                (post and i == 0)
                or (
                    not post
                    and i in (1, 2)
                    and self.data_loading == "jet_const"
                )
            )

            ax.hist(
                orig_feat.numpy(),
                bins=bins,
                density=True,
                log=log_scale,
                color=color[i],
                label="Original",
                alpha=0.7
            )

            ax.hist(
                reco_feat.numpy(),
                bins=bins,
                density=True,
                log=log_scale,
                color="purple",
                label="Reconstructed",
                alpha=0.9,
                histtype="step"
            )

            ax.set_xlabel(feat_name)
            ax.set_ylabel("Density")
            ax.legend()

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        stage = "postprocessed" if post else "preprocessed"

        fig.suptitle(
            f"Test - {stage} - "
            f"{self.model_name}-VQVAE-"
            f"rot:{self.rotation}-"
            f"cb:{self.cb_size}"
        )

        fig.tight_layout()

        return fig

    # ---------------------------------------------------------
    # CODEBOOK USAGE
    # ---------------------------------------------------------

    def _create_cb_usage(self, idx):
        """Create codebook usage plot."""

        cb_usage = (
            len(torch.unique(idx))
            / int(self.cb_size)
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        bins = (
            np.arange(self.cb_size + 1)
            - 0.5
        )

        ax.hist(
            idx.numpy(),
            density=True,
            bins=bins,
            color="brown",
            alpha=0.8
        )

        ax.set_xlim(
            -0.5,
            self.cb_size - 0.5
        )

        ax.set_xlabel(
            f"Quantization index "
            f"(CB-usage={cb_usage:.4f})"
        )

        ax.set_ylabel("Density")

        fig.suptitle(
            f"Test - {self.model_name}-VQVAE-"
            f"rot:{self.rotation}-"
            f"cb:{self.cb_size}"
        )

        fig.tight_layout()

        return fig
