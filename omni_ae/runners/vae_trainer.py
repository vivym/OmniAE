import copy

import torch
import torch.nn as nn
from accelerate import Accelerator
from ray.train import CheckpointConfig, ScalingConfig, SyncConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from omni_ae.models import AutoencoderKL
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

        vae, ema_vae, params_to_optimize = self.setup_models(accelerator)

        optimizer, lr_scheduler = self.setup_optimizer(params_to_optimize)

        train_dataloader, vae, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, vae, optimizer, lr_scheduler
        )
        train_dataloader: DataLoader
        vae: AutoencoderKL
        optimizer: torch.optim.Optimizer
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

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

        progress_bar = tqdm(
            range(0, self.runner_config.max_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        # TODO: skip initial steps if resuming from checkpoint

        # done = False
        # while not done:
        #     vae.train()

        #     train_loss = 0.0

        #     for step, batch in enumerate(train_dataloader):
        #         with accelerator.accumulate(vae):
        #             loss: torch.Tensor = self.training_step(
        #                 batch,
        #                 unet=unet,
        #                 vae=vae,
        #                 text_encoders=text_encoder,
        #                 noise_scheduler=noise_scheduler,
        #                 weight_dtype=weight_dtype,
        #             )

        #             # Gather the losses across all processes for logging (if we use distributed training).
        #             losses: torch.Tensor = accelerator.gather(
        #                 loss.repeat(self.runner_config.train_batch_size)
        #             )
        #             train_loss += losses.mean()

        #             # Backpropagate
        #             accelerator.backward(loss)
        #             if accelerator.sync_gradients:
        #                 accelerator.clip_grad_norm_(
        #                     params_to_optimize, self.runner_config.gradient_clipping
        #                 )

        #             optimizer.step()
        #             lr_scheduler.step()
        #             optimizer.zero_grad()

        #         logs = {"lr": lr_scheduler.get_last_lr()[0]}
        #         progress_bar.set_postfix(**logs)

        #         # Checks if the accelerator has performed an optimization step behind the scenes
        #         if accelerator.sync_gradients:
        #             if self.runner_config.use_ema:
        #                 ema_vae.step(vae.parameters())

        #             progress_bar.update(1)
        #             global_step += 1
        #             train_loss = train_loss / self.runner_config.gradient_accumulation_steps
        #             accelerator.log({"train_loss": train_loss}, step=global_step)
        #             train_loss = 0.0

        #         if global_step >= self.runner_config.max_steps:
        #             done = True
        #             break

        accelerator.end_training()

    def setup_models(
        self, accelerator: Accelerator
    ) -> tuple[AutoencoderKL, EMAModel | None, list[torch.nn.Parameter]]:
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
        else:
            ema_vae = None

        params_to_optimize = vae.parameters()

        if self.runner_config.use_ema:
            ema_vae.to(accelerator.device)

        return vae, ema_vae, params_to_optimize
