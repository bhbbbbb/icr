from typing import Tuple, Iterable

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_utils import BaseModelUtils
from model_utils.criteria import Loss, Criteria

from .config import ICRModelUtilsConfig
from .loss import BlancedLogLoss
# from ..dataset import ICRDataset

Criteria.register_criterion(Loss)

class ICRModelUtils(BaseModelUtils[DataLoader, DataLoader]):

    config: ICRModelUtilsConfig
    loss_fn: BlancedLogLoss

    def _init(self):
        self.loss_fn = BlancedLogLoss(self.config.loss_class_weights, self.config.device)
        return

    @staticmethod
    def _get_optimizer(model, config: ICRModelUtilsConfig) -> Adam:
        return Adam(
            model.parameters(),
            lr=config.learning_rate, # < with OneCycleLR this has no actual effect
            weight_decay=config.weight_decay,
        )

    @staticmethod
    def _get_scheduler(optimizer: Adam, config: ICRModelUtilsConfig) -> OneCycleLR:
        return OneCycleLR(
            optimizer=optimizer,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
        )

    def _train_epoch(self, train_data: DataLoader):

        self.model.train()

        train_loss = .0

        pbar = tqdm(
            train_data,
            disable=not self.config.progress_bar,
        )
        pbar: Iterable[Tuple[Tensor, Tensor]]

        step = 0

        for features, labels in pbar:

            self.optimizer.zero_grad()

            features = features.to(torch.float).to(self.config.device)
            labels = labels.to(torch.long).to(self.config.device)

            predictions: Tensor = self.model(features)
            predictions = torch.sigmoid(predictions)
            loss: Tensor = self.loss_fn.forward(labels, predictions.squeeze(1))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.config.device == 'cuda:0':
                torch.cuda.synchronize()

            running_loss = loss.item()
            train_loss += running_loss

            lr = self.scheduler.get_last_lr()
            lr = sum(lr) / len(lr)

            pbar.set_description(f'LR: {lr: .4e} Running Loss: {running_loss:.6f}')

            step += 1

        train_loss /= step
        if self.config.device:
            torch.cuda.empty_cache()
        return Loss(train_loss)

    @torch.no_grad()
    def _eval_epoch(self, eval_data: DataLoader):

        self.model.eval()

        eval_loss = .0

        pbar = tqdm(
            eval_data,
            disable=not self.config.progress_bar,
        )
        pbar: Iterable[Tuple[Tensor, Tensor]]

        step = 0

        for features, labels in pbar:

            features = features.to(torch.float).to(self.config.device)
            labels = labels.to(torch.long).to(self.config.device)

            predictions: Tensor = self.model(features)
            predictions = torch.sigmoid(predictions)
            loss: Tensor = self.loss_fn.forward(labels, predictions.squeeze(1))

            running_loss = loss.item()
            eval_loss += running_loss

            pbar.set_description(f'Running Loss: {running_loss:.6f}')
            step += 1

        eval_loss /= step
        return Loss(eval_loss)
