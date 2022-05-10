################################################################################
#
# Implementation of a prototype model for speaker recognition.
#
# It consists of a single CNN layer which convolutes the input spectrogram,
# a pooling layer which reduces the CNN output to a fixed-size speaker embedding,
# and a single FC classification layer.
#
# Author(s): Nik Vaessen
################################################################################

from collections import deque
from typing import List, Optional, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from skeleton.data.batch import HIDBatch
from skeleton.evaluation.evaluator import (
    EmbeddingSample,
    EvaluationPair,
    SpeakerRecognitionEvaluator,
)
from skeleton.layers.arcface import ArcMarginProduct, HotelIdModel
from skeleton.layers.resnet import ResNet34

################################################################################
# Implement the lightning module for training a prototype model
# for speaker recognition on tiny-voxceleb


class HotelID(LightningModule):
    def __init__(
        self,
        num_embedding: int,
        num_hotels: int,
        width: int,
        height: int,
        learning_rate: float,
        num_epochs: int = 120,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        min_lr: float = 0.0,
        epochs: int = 50,
        device="cuda:0",
    ):
        super().__init__()

        # hyperparameters
        self.epochs = epochs
        # self.num_inp_features = num_inp_features
        self.num_embedding = num_embedding
        self.num_hotels = num_hotels
        self.width = width
        self.height = height
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.min_lr = min_lr

        # evaluation data
        self.evaluator = SpeakerRecognitionEvaluator()

        # Embedding layer
        backbone_name="eca_nfnet_l0"
        self.embedding_layer = HotelIdModel(self.num_hotels, self.num_embedding, backbone_name)

        # Fully-connected layer
        self.prediction_layer = ArcMarginProduct(self.num_embedding, self.num_hotels, s=30.0, m=0.20, easy_margin=False, device=device)

        # The loss function. 
        # self.loss_fn = F.nll_loss
        self.loss_fn = F.cross_entropy

        # used to keep track of training/validation accuracy
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # save hyperparameters for easy reloading of model
        self.save_hyperparameters()

    def forward(self, images: t.Tensor, labels: t.Tensor = None) -> t.Tensor:
        embedding = self.compute_embedding(images, labels)
        if labels is not None:
            prediction = self.compute_prediction(embedding, labels)
            return embedding, prediction
        return embedding

    def compute_embedding(self, images: t.Tensor, labels: t.Tensor) -> t.Tensor:
        return self.embedding_layer(images, targets=labels)

    def compute_prediction(self, embedding: t.Tensor, labels: t.Tensor) -> t.Tensor:
        return self.prediction_layer(embedding, labels)
         
    def training_step(
        self, batch: HIDBatch, *args, **kwargs
    ) -> t.Tensor:
        # first unwrap the batch into the input tensor and ground truth labels
        assert isinstance(batch, HIDBatch)
        # The 3 is for colors
        assert list(batch.images.shape) == [batch.batch_size, 3, self.width, self.height]
        assert len(batch.images.shape) == 4

        images = batch.images
        labels = batch.labels

        # then compute the forward pass
        embedding, prediction = self.forward(images, labels)

        # based on the output of the forward pass we compute the loss
        loss = self.loss_fn(prediction, labels)

        # based on the output of the forward pass we compute some metrics
        self.train_acc(prediction, labels)

        # log training loss
        self.log("loss", loss, prog_bar=False)

        # store recent training embeddings
        # self._add_batch_to_embedding_queue(embedding)

        # The value we return will be minimized
        return loss

    def training_epoch_end(self, outputs: List[t.Tensor]) -> None:
        # at the end of a training epoch we log our metrics
        self.log("train_acc", self.train_acc, prog_bar=True)

        # update the mean&std of training embeddings
        # mean, std = self.compute_mean_std_batch([e for e in self.embeddings_queue])

        # with t.no_grad():
        #     mean = mean.to(self.mean_embedding.device)
        #     std = std.to(self.std_embedding.device)

        #     self.mean_embedding[:] = mean
        #     self.std_embedding[:] = std

    def validation_step(
        self, batch: HIDBatch, *args, **kwargs
    ) -> Tuple[t.Tensor, t.Tensor, List[str]]:
        # first unwrap the batch into the input tensor and ground truth labels
        assert isinstance(batch, HIDBatch)
        # The 3 is for colors
        assert list(batch.images.shape) == [batch.batch_size, 3, self.width, self.height]
        assert len(batch.images.shape) == 4

        images = batch.images
        labels = batch.labels
        sample_keys = batch.image_ids

        # then compute the forward pass
        embedding, prediction = self.forward(images, labels)

        # based on the output of the forward pass we compute the loss
        loss = self.loss_fn(prediction, labels)

        # based on the output of the forward pass we compute some metrics
        self.val_acc(prediction, labels)

        # the value(s) we return will be saved until the end op the epoch
        # and passed to `validation_epoch_end`
        return loss

    def validation_epoch_end(
        self, loss: t.Tensor
    ) -> None:
        # at the end of a validation epoch we compute the validation EER
        # based on the embeddings and log all metrics

        # unwrap outputs
        # embeddings = [embedding for embedding, _, _ in outputs]
        # losses = [loss for _, loss, _ in outputs]
        # sample_keys = [key for _, _, key in outputs]

        # log metrics
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_loss", t.mean(t.stack(loss)), prog_bar=True)



    def configure_optimizers(self):
        # setup the optimization algorithm
        optimizer = t.optim.SGD(
            self.parameters(),
            momentum=self.momentum,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # setup the learning rate schedule.
        schedule = {
            # Required: the scheduler instance.
            "scheduler": t.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=self.min_lr
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after an optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return [optimizer], [schedule]

    """
    def _evaluate_embeddings(
        self,
        embeddings: List[t.Tensor],
        sample_keys: List[List[str]],
        pairs: List[EvaluationPair],
    ):
        assert len(embeddings) == len(sample_keys)

        embedding_list: List[EmbeddingSample] = []

        for embedding_tensor, key_list in zip(embeddings, sample_keys):
            if len(key_list) != embedding_tensor.shape[0]:
                raise ValueError("batch dimension is missing or incorrect")

            for idx, sample_id in enumerate(key_list):
                embedding_list.append(
                    EmbeddingSample(
                        sample_id, embedding_tensor[idx, :].squeeze().to("cpu")
                    )
                )

        result = self.evaluator.evaluate(
            pairs,
            embedding_list,
            None,
            mean_embedding=self.mean_embedding.to("cpu")
            if self.apply_embedding_centering
            else None,
            std_embedding=self.std_embedding.to("cpu")
            if self.apply_embedding_centering
            else None,
            length_normalize=self.apply_embedding_length_norm,
        )

        return result["eer"]

    def _add_batch_to_embedding_queue(self, embedding: t.Tensor):
        # make sure to keep it into CPU memory
        embedding = embedding.detach().to("cpu")

        # unbind embedding of shape [BATCH_SIZE, EMBEDDING_SIZE] into a list of
        # tensors of shape [EMBEDDING_SIZE] with len=BATCH_SIZE
        embedding_list = t.unbind(embedding, dim=0)

        self.embeddings_queue.extend([embedding for embedding in embedding_list])

    @staticmethod
    def compute_mean_std_batch(all_tensors: List[t.Tensor]):
        # compute mean and std over each dimension of EMBEDDING_SIZE
        # with a tensor of shape [NUM_SAMPLES, EMBEDDING_SIZE]
        stacked_tensors = t.stack(all_tensors)

        std, mean = t.std_mean(stacked_tensors, dim=0)

        return mean, std
    """
