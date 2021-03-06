import numpy as np
import torch
from base import BaseTrainer
from tqdm import tqdm
from utils.util import dummy_context_mgr


class Trainer(BaseTrainer):
    """Trainer class.

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        resume,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        train_logger=None,
        comet_exp=None,
    ):
        super(Trainer, self).__init__(
            model, loss, metrics, optimizer, resume, config, train_logger, comet_exp
        )
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        with self.experiment.train() if self.experiment is not None else dummy_context_mgr():
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                for metric in self.metrics:
                    metric.update(output, target)

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            epoch,
                            batch_idx * self.data_loader.batch_size,
                            self.data_loader.n_samples,
                            100.0 * batch_idx / len(self.data_loader),
                            loss.item(),
                        )
                    )
                    # fig = make_grid(data.cpu(), nrow=8, normalize=True)
                    # plt.imshow(np.transpose(fig.numpy(), (1, 2, 0)), interpolation='nearest')
                    # self.experiment.log_figure('input')

            # finalize metrics
            for metric in self.metrics:
                metric.eval()

            log = {'loss': total_loss / len(self.data_loader), 'metrics': []}
            if self.experiment is not None:
                self.experiment.log_metric('epoch_loss', log['loss'], step=epoch)
                for m in self.metrics:
                    self.experiment.log_metric(m.name, m.value, step=epoch)

        if self.do_validation:
            with self.experiment.test() if self.experiment is not None else dummy_context_mgr():
                val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.experiment is not None:
            self.experiment.log_epoch_end(self, epoch)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                total_val_loss += loss.item()

                for metric in self.metrics:
                    metric.update(output, target)
                # fig = make_grid(data.cpu(), nrow=8, normalize=True)
                # plt.imshow(np.transpose(fig.numpy(), (1, 2, 0)), interpolation='nearest')
                # self.experiment.log_figure('input')

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        # finalize metrics
        for metric in self.metrics:
            metric.eval()

        log = {'val_loss': total_val_loss / len(self.valid_data_loader), 'val_metrics': []}

        if self.experiment is not None:
            self.experiment.log_metric('epoch_loss', log['val_loss'], step=epoch)
            for m in self.metrics:
                self.experiment.log_metric(m.name, m.value, step=epoch)

        return log
