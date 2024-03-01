import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from ray.train import CheckpointConfig, ScalingConfig, SyncConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from omni_ae.models import AutoencoderKL, Discriminator, LPIPSMetric
from omni_ae.utils import logging
from omni_ae.utils.ema import EMAModel
from .runner import Runner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VAETrainer(Runner):
    def __call__(self):
        trainer = TorchTrainer(
            self.train_loop_per_worker,
            scaling_config=ScalingConfig(
                trainer_resources={"CPU": self.runner_config.num_cpus_per_worker},
                num_workers=self.runner_config.num_devices,
                use_gpu=self.runner_config.num_gpus_per_worker > 0,
                resources_per_worker={
                    "CPU": self.runner_config.num_cpus_per_worker,
                    "GPU": self.runner_config.num_gpus_per_worker,
                },
            ),
            run_config=RunConfig(
                name=self.runner_config.name,
                storage_path=self.runner_config.storage_path,
                # TODO: failure_config
                checkpoint_config=CheckpointConfig(),
                sync_config=SyncConfig(
                    sync_artifacts=True,
                ),
                verbose=self.runner_config.verbose_mode,
                log_to_file=True,
            ),
        )

        trainer.fit()

    def train_loop_per_worker(self):
        if self.runner_config.allow_tf32:
            torch.set_float32_matmul_precision("high")

        accelerator = self.setup_accelerator()

        train_dataloader = self.setup_dataloader()

        vae, ema_vae, lpips_metric, logvar, discriminator = self.setup_models(accelerator)
        params_to_optimize = list(vae.parameters()) + [logvar] + list(discriminator.parameters())

        optimizer, lr_scheduler = self.setup_optimizer(vae.parameters())

        disc_optimizer, disc_lr_scheduler = self.setup_optimizer(discriminator.parameters())

        (
            train_dataloader, vae, logvar, discriminator, optimizer, lr_scheduler, disc_optimizer, disc_lr_scheduler
        ) = accelerator.prepare(
            train_dataloader, vae, logvar, discriminator, optimizer, lr_scheduler, disc_optimizer, disc_lr_scheduler
        )
        train_dataloader: DataLoader
        vae: AutoencoderKL
        logvar: nn.Parameter
        discriminator: Discriminator
        optimizer: torch.optim.Optimizer
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        disc_optimizer: torch.optim.Optimizer
        disc_lr_scheduler: torch.optim.lr_scheduler.LRScheduler

        total_batch_size = (
            self.runner_config.train_batch_size * accelerator.num_processes * self.runner_config.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.runner_config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.runner_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.runner_config.max_steps}")
        global_step = 0
        initial_global_step = 0
        gan_stage = "none" if self.runner_config.discriminator_start_steps > global_step else "generator"

        progress_bar = tqdm(
            range(0, self.runner_config.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        # TODO: skip initial steps if resuming from checkpoint

        done = False
        while not done:
            vae.train()

            loss_dict: dict[str, torch.Tensor] = {}

            for batch in train_dataloader:
                with accelerator.accumulate(vae, logvar, discriminator):
                    loss, loss_dict_i = self.training_step(
                        batch,
                        vae=vae,
                        lpips_metric=lpips_metric,
                        logvar=logvar,
                        discriminator=discriminator,
                        gan_stage=gan_stage,
                    )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    for k, v in loss_dict_i.items():
                        if k not in loss_dict:
                            loss_dict[k] = 0.0

                        losses: torch.Tensor = accelerator.gather(v)
                        loss_dict[k] += losses.mean()

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            params_to_optimize, self.runner_config.gradient_clipping
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad()

                logs = {"lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.runner_config.use_ema:
                        ema_vae.step(vae.parameters())

                    progress_bar.update(1)
                    global_step += 1

                    for k, v in loss_dict.items():
                        loss_dict[k] = v.cpu() / self.runner_config.gradient_accumulation_steps
                    accelerator.log(loss_dict, step=global_step)
                    loss_dict: dict[str, torch.Tensor] = {}

                    if gan_stage == "generator":
                        gan_stage = "discriminator"
                    elif gan_stage == "discriminator":
                        gan_stage = "generator"

                    if gan_stage == "none" and global_step >= self.runner_config.discriminator_start_steps:
                        gan_stage = "generator"

                if global_step >= self.runner_config.max_steps:
                    done = True
                    break

        accelerator.end_training()

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        vae : AutoencoderKL,
        lpips_metric: LPIPSMetric,
        logvar: nn.Parameter,
        discriminator: Discriminator,
        gan_stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        samples = batch["pixel_values"]

        posterior = vae.encode(samples).latent_dist

        z = posterior.sample()

        rec_samples = vae.decode(z).sample

        loss_rec = F.l1_loss(samples, rec_samples)

        loss_perceptual = lpips_metric(samples, rec_samples)

        reconstruction_loss_weight = self.runner_config.reconstruction_loss_weight
        perceptual_loss_weight = self.runner_config.perceptual_loss_weight
        loss_nll = (
            reconstruction_loss_weight * loss_rec + perceptual_loss_weight * loss_perceptual
        ) / torch.exp(logvar) + logvar
        loss_nll = loss_nll.sum() / loss_nll.shape[0]

        loss_kl = posterior.kl()
        loss_kl = loss_kl.sum() / loss_kl.shape[0]

        nll_loss_weight = self.runner_config.nll_loss_weight
        kl_loss_weight = self.runner_config.kl_loss_weight
        loss = nll_loss_weight * loss_nll + kl_loss_weight * loss_kl

        loss_dict = {
            "loss_rec": loss_rec.detach(),
            "loss_perceptual": loss_perceptual.detach(),
            "loss_nll": loss_nll.detach(),
            "loss_kl": loss_kl.detach(),
        }

        if gan_stage == "generator":
            # Update the Generator
            logits_fake = discriminator(rec_samples)
            loss_g = -torch.mean(logits_fake)

            last_layer_weight = vae.decoder.conv_out.conv.weight
            disc_weight = compute_adaptive_disc_weight(
                loss_nll, loss_g, last_layer_weight
            ) * self.runner_config.discriminator_loss_weight

            loss += disc_weight * loss_g

            loss_dict["loss_g"] = loss_g.detach()
            loss_dict["disc_weight"] = disc_weight
        elif gan_stage == "discriminator":
            # Update the Discriminator
            logits_real = discriminator(samples)
            logits_fake = discriminator(rec_samples.detach())

            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            loss_d = (loss_real + loss_fake) * 0.5

            loss += loss_d # TODO: add weight to loss_d

            loss_dict["loss_real"] = loss_real.detach()
            loss_dict["loss_fake"] = loss_fake.detach()
            loss_dict["loss_d"] = loss_d.detach()

        loss_dict["loss"] = loss.detach()

        return loss, loss_dict

    def setup_models(
        self, accelerator: Accelerator
    ) -> tuple[AutoencoderKL, EMAModel | None, LPIPSMetric, nn.Parameter, Discriminator]:
        vae = AutoencoderKL(
            in_channels=self.model_config.in_channels,
            out_channels=self.model_config.out_channels,
            down_block_types=self.model_config.down_block_types,
            up_block_types=self.model_config.up_block_types,
            block_out_channels=self.model_config.block_out_channels,
            use_gc_blocks=self.model_config.use_gc_blocks,
            mid_block_type=self.model_config.mid_block_type,
            mid_block_use_attention=self.model_config.mid_block_use_attention,
            mid_block_num_attention_heads=self.model_config.mid_block_num_attention_heads,
            layers_per_block=self.model_config.layers_per_block,
            act_fn=self.model_config.act_fn,
            num_attention_heads=self.model_config.num_attention_heads,
            latent_channels=self.model_config.latent_channels,
            norm_num_groups=self.model_config.norm_num_groups,
            scaling_factor=self.model_config.scaling_factor,
        )

        vae.train()

        # Create EMA for the vae.
        if self.runner_config.use_ema:
            ema_vae = copy.deepcopy(vae)
            ema_vae = EMAModel(
                ema_vae.parameters(),
            )
            ema_vae.to(accelerator.device)
        else:
            ema_vae = None

        lpips_metric = LPIPSMetric.from_pretrained(self.runner_config.lpips_model_name_or_path)
        lpips_metric.to(accelerator.device)

        logvar = nn.Parameter(torch.full((), self.runner_config.init_logvar, dtype=torch.float32))

        discriminator = Discriminator()

        return vae, ema_vae, lpips_metric, logvar, discriminator


def compute_adaptive_disc_weight(
    loss_nll: torch.Tensor, loss_g: torch.Tensor, last_layer_weight: nn.Parameter
) -> torch.Tensor:
    nll_grads = torch.autograd.grad(loss_nll, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]

    with torch.no_grad():
        disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        disc_weight = torch.clamp(disc_weight, min=0.0, max=1e4)

    return disc_weight
