from typing import Tuple, Iterable

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_utils import BaseModelUtils
from model_utils.criteria import Loss, Criteria, Accuarcy

from .icr_model_utils_config import ICRModelUtilsConfig
from .loss import BlancedLogLoss
# from ..dataset import ICRDataset

Criteria.register_criterion(Loss)
Criteria.register_criterion(Accuarcy)
Recall = Criteria.register_criterion(
    'Recall',
    name='Recall',
    primary=False,
    full_name='Recall'
)(Accuarcy)
Precision = Criteria.register_criterion(
    'Precision',
    name='precision',
    primary=False,
    full_name='Precision'
)(Accuarcy)

class ICRModelUtils(BaseModelUtils[DataLoader, DataLoader]):

    config: ICRModelUtilsConfig
    train_loss_fn: BlancedLogLoss
    eval_loss_fn: BlancedLogLoss

    def _init(self):
        self.train_loss_fn = BlancedLogLoss(
            self.config.loss_class_weights, self.config.device)
        self.eval_loss_fn = BlancedLogLoss([1., 1.], self.config.device)
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
        # n_class1 = 0

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
            loss: Tensor = self.train_loss_fn.forward(labels, predictions.squeeze(1))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.config.device == 'cuda:0':
                torch.cuda.synchronize()

            running_loss = loss.item()
            # n_class1 += (labels == 1).sum()
            train_loss += running_loss

            lr = self.scheduler.get_last_lr()
            lr = sum(lr) / len(lr)

            pbar.set_description(f'LR: {lr: .4e} Running Loss: {running_loss:.6f}')

            step += 1

        train_loss /= step
        if self.config.device:
            torch.cuda.empty_cache()
        return (
            Loss(train_loss),
            # Class1Ratio(n_class1 / (train_data.batch_size * step)),
        )

    @torch.no_grad()
    def _eval_epoch(self, eval_data: DataLoader):

        self.model.eval()

        eval_loss = .0
        n_samples = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

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
            predictions = torch.sigmoid(predictions).squeeze(1)
            loss: Tensor = self.eval_loss_fn.forward(labels, predictions)

            running_loss = loss.item()
            eval_loss += running_loss
            tp += ((predictions > .5) * (labels == 1)).sum().item()
            tn += ((predictions <= .5) * (labels == 0)).sum().item()
            fp += ((predictions > .5) * (labels == 0)).sum().item()
            fn += ((predictions <= .5) * (labels == 1)).sum().item()
            # n_correct += ((predictions > .5) == labels).sum().item()
            # eval_n_class1 += (labels == 1).sum().item()
            n_samples += len(labels)

            pbar.set_description(f'Running Loss: {running_loss:.6f}')
            step += 1

        eval_loss /= step
        n_correct = tp + tn
        return (
            Loss(eval_loss),
            Accuarcy(n_correct / n_samples),
            Precision(tp / (tp + fp + 1e-10)),
            Recall(tp / (tp + fn + 1e-10)),
        )
