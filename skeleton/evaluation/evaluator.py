################################################################################
#
# Implement an Evaluator object which encapsulates the process
# computing performance metric of speaker recognition task.
#
# Author(s): Nik Vaessen
################################################################################

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch as t
from torch.nn import CosineSimilarity
from torch.nn.functional import normalize
from tqdm import tqdm

# from skeleton.evaluation.eval_metrics import calculate_eer

################################################################################
# define data structures required for evaluating


@dataclass
class EvaluationPair:
    same_speaker: bool
    sample1_id: str
    sample2_id: str


@dataclass
class EmbeddingSample:
    sample_id: str
    embedding: Union[t.Tensor, List[t.Tensor]]

    def __post_init__(self):
        if isinstance(self.embedding, list):
            [self._verify_embedding(e) for e in self.embedding]
        elif isinstance(self.embedding, t.Tensor):
            self._verify_embedding(self.embedding)
        else:
            raise ValueError(f"unexpected {type(self.embedding)=}")

    @staticmethod
    def _verify_embedding(embedding: t.Tensor):
        if len(embedding.shape) != 1:
            raise ValueError("expected embedding to be 1-dimensional tensor")


################################################################################
# implementation of cosine-distance evaluator


class CosineDistanceSimilarityModule(t.nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_distance = CosineSimilarity()

    def forward(self, embedding: t.Tensor, other_embedding: t.Tensor):
        # we assume both inputs have dimensionality [batch_size, NUM_FEATURES]
        cos_dist = self.cosine_distance(embedding, other_embedding)

        # return a score between [-1, 1]
        return cos_dist


class SpeakerRecognitionEvaluator:
    @classmethod
    def evaluate(
        cls,
        pairs: List[EvaluationPair],
        samples: List[EmbeddingSample],
        cohort: List[t.Tensor] = None,
        mean_embedding: Optional[t.Tensor] = None,
        std_embedding: Optional[t.Tensor] = None,
        length_normalize: bool = False,
        skip_eer: bool = False,
    ):
        # create a hashmap for quicker access to samples based on key
        sample_map = {}

        for sample in samples:
            if sample.sample_id in sample_map:
                raise ValueError(f"duplicate key {sample.sample_id}")

            sample_map[sample.sample_id] = sample

        # compute a list of ground truth scores and prediction scores
        ground_truth_scores = []
        prediction_pairs = []

        for pair in pairs:
            if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
                warn(f"{pair.sample1_id} or {pair.sample2_id} not in sample_map")
                return {
                    "eer": -1,
                }

            s1 = sample_map[pair.sample1_id]
            s2 = sample_map[pair.sample2_id]

            gt = 1 if pair.same_speaker else 0

            ground_truth_scores.append(gt)
            prediction_pairs.append((s1, s2))

        if cohort is not None:
            prediction_scores = cls._compute_asnorm_prediction_scores(
                prediction_pairs,
                cohort,
                length_normalize=length_normalize,
                mean_embedding=mean_embedding,
                std_embedding=std_embedding,
            )
        else:
            prediction_scores = cls._compute_prediction_scores(
                prediction_pairs,
                length_normalize=length_normalize,
                mean_embedding=mean_embedding,
                std_embedding=std_embedding,
            )

        # normalize scores to be between 0 and 1
        prediction_scores = np.clip((np.array(prediction_scores) + 1) / 2, 0, 1)
        prediction_scores = prediction_scores.tolist()

        if skip_eer:
            return prediction_scores

        # compute EER
        try:
            eer = calculate_eer(ground_truth_scores, prediction_scores)
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that programs relying on result don't crash
            print(f"EER calculation had {e}")
            eer = 1

        return {
            "eer": eer,
        }

    @classmethod
    def _compute_asnorm_prediction_scores(
        cls,
        pairs: List[Tuple[EmbeddingSample, EmbeddingSample]],
        cohort: t.Tensor,
        length_normalize: bool = False,
        mean_embedding: Optional[t.Tensor] = None,
        std_embedding: Optional[t.Tensor] = None,
    ) -> List[float]:
        cohort_size = 32
        asnorm_scores = []
        cosine_sim = CosineDistanceSimilarityModule()
        for enrollment, test in tqdm(pairs, desc="Computing ASNorm prediction scores"):
            # Extract the embedding only
            enrollment = enrollment.embedding
            test = test.embedding

            # Optionally normalize by subtracting mean and dividing by std
            if mean_embedding is not None and std_embedding is not None:
                enrollment = (enrollment - mean_embedding) / (std_embedding + 1e-12)
                test = (test - mean_embedding) / (std_embedding + 1e-12)

            # Optionally normalize the length
            if length_normalize:
                enrollment = length_norm_batch(enrollment)
                test = length_norm_batch(test)

            # Convert [EMBEDDING_SIZE] to [1, EMBEDDING_SIZE]
            enrollment = enrollment.reshape(1, enrollment.shape[0])
            # Find the `cohort_size` most similar embeddings for `enrollment`
            dist = cosine_sim(enrollment, cohort)
            values, _ = t.topk(dist, cohort_size)
            # cohort_e_top = cohort[indices]
            score_e_cohort_e_top = values
            std_score_e_cohort_e_top, mean_score_e_cohort_e_top = t.std_mean(
                score_e_cohort_e_top
            )

            # Convert [EMBEDDING_SIZE] to [1, EMBEDDING_SIZE]
            test = test.reshape(1, test.shape[0])
            # Find the `cohort_size` most similar embeddings for `enrollment`
            dist = cosine_sim(test, cohort)
            values, _ = t.topk(dist, cohort_size)
            # cohort_t_top = cohort[indices]
            score_t_cohort_t_top = values
            std_score_t_cohort_t_top, mean_score_t_cohort_t_top = t.std_mean(
                score_t_cohort_t_top
            )

            score = cosine_sim(enrollment, test)[0]
            e_norm = (score - mean_score_e_cohort_e_top) / std_score_e_cohort_e_top
            t_norm = (score - mean_score_t_cohort_t_top) / std_score_t_cohort_t_top

            asnorm_scores.append(0.5 * (e_norm + t_norm))

        return t.stack(asnorm_scores)

    @classmethod
    def _compute_prediction_scores(
        cls,
        pairs: List[Tuple[EmbeddingSample, EmbeddingSample]],
        length_normalize: bool = False,
        mean_embedding: Optional[t.Tensor] = None,
        std_embedding: Optional[t.Tensor] = None,
    ) -> List[float]:
        left_samples, right_samples = cls._transform_pairs_to_tensor(pairs)

        if mean_embedding is not None and std_embedding is not None:
            left_samples = center_batch(left_samples, mean_embedding, std_embedding)
            right_samples = center_batch(right_samples, mean_embedding, std_embedding)

        if length_normalize:
            left_samples = length_norm_batch(left_samples)
            right_samples = length_norm_batch(right_samples)

        scores = cls._compute_cosine_scores(left_samples, right_samples)

        return scores

    @staticmethod
    def _compute_cosine_scores(
        left_samples: t.Tensor, right_samples: t.Tensor
    ) -> List[float]:
        # compute the scores
        score_tensor = CosineDistanceSimilarityModule()(left_samples, right_samples)

        return score_tensor.detach().cpu().numpy().tolist()

    @classmethod
    def _transform_pairs_to_tensor(
        cls, pairs: List[Tuple[EmbeddingSample, EmbeddingSample]]
    ):
        # construct the comparison batches
        b1 = []
        b2 = []

        for s1, s2 in pairs:
            b1.append(s1.embedding)
            b2.append(s2.embedding)

        b1 = t.stack(b1)
        b2 = t.stack(b2)

        return b1, b2


################################################################################
# Utility methods common for evaluating


def center_batch(embedding_tensor: t.Tensor, mean: t.Tensor, std: t.Tensor):
    # center the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    # using the computed mean and std
    centered = (embedding_tensor - mean) / (std + 1e-12)

    return centered


def length_norm_batch(embedding_tensor: t.Tensor):
    # length normalize the batch with shape [NUM_PAIRS, EMBEDDING_SIZE]
    return normalize(embedding_tensor, dim=1)
