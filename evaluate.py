# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import gin
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from arg_parser import parse_arguments, TaskMode
from model.dataset.isaac_sim_dataset import XMobilityIsaacSimDataModule
from model.trainer import XMobilityTrainer
from model.eval.prediction_evaluator import PredictionEvaulator


@gin.configurable
def evaluate_observation(dataset_path, checkpoint_path, wandb_entity_name,
                         wandb_project_name, wandb_run_name, num_gpus,
                         precision):
    data = XMobilityIsaacSimDataModule(dataset_path=dataset_path)
    model = XMobilityTrainer.load_from_checkpoint(checkpoint_path)
    model.eval()

    wandb_logger = WandbLogger(entity=wandb_entity_name,
                               project=wandb_project_name,
                               name=wandb_run_name,
                               group="DDP",
                               log_model=True)

    trainer = pl.Trainer(num_nodes=num_gpus,
                         precision=precision,
                         logger=wandb_logger,
                         strategy='ddp')
    trainer.test(model, datamodule=data)

    wandb.finish()


@gin.configurable
def evaluate_prediction(dataset_path,
                        checkpoint_path,
                        wandb_entity_name,
                        wandb_project_name,
                        wandb_run_name,
                        max_history_length=2,
                        max_future_length=[1, 3, 6],
                        use_trained_policy=False):
    data_module = XMobilityIsaacSimDataModule(
        dataset_path=dataset_path,
        sequence_length=max_history_length + np.max(max_future_length))
    model = XMobilityTrainer.load_from_checkpoint(checkpoint_path,
                                                  strict=False)
    model.eval()

    wandb_logger = WandbLogger(entity=wandb_entity_name,
                               project=wandb_project_name,
                               name=wandb_run_name,
                               group="DDP",
                               log_model=False)

    evaulator = PredictionEvaulator(model, data_module, wandb_logger,
                                    max_history_length, max_future_length,
                                    use_trained_policy)
    evaulator.compute()

    wandb.finish()


def main():
    args = parse_arguments(TaskMode.EVAL)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    if args.eval_target == 'observation':
        # Run the evaluation loop.
        evaluate_observation(args.dataset_path, args.checkpoint_path,
                             args.wandb_entity_name, args.wandb_project_name,
                             args.wandb_run_name)
    elif args.eval_target == 'imagination':
        evaluate_prediction(args.dataset_path, args.checkpoint_path,
                            args.wandb_entity_name, args.wandb_project_name,
                            args.wandb_run_name)
    else:
        raise ValueError('Unsupported eval target.')


if __name__ == '__main__':
    main()
