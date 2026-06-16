from itertools import zip_longest
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from . import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def shuffle_combinations(iterable: Iterable, replacement: bool = True) -> Generator:
    """Generates shuffled pair combinations for any iterable data provided.

    Args:
        iterable: data to generate pair combinations from
        replacement: enable to include combinations of same samples,
            equivalent to itertools.combinations_with_replacement

    Returns:
        Generator of shuffled pairs as a tuple
    """
    n = len(iterable)
    k = 1 if not replacement else 0
    idxs = np.stack(np.triu_indices(n, k), axis=-1)
    for i in np.random.RandomState(seed=42).permutation(len(idxs)):
        _idx, idx = idxs[i, :]
        yield iterable[_idx], iterable[idx]


class ContrastiveDataset(IterableDataset):
    def __init__(
        self,
        sentences: List[str],
        labels: List[Union[int, float]],
        multilabel: bool,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        """Generates positive and negative text pairs for contrastive learning.

        Args:
            sentences (List[str]): text sentences to generate pairs from
            labels (List[Union[int, float]]): labels for each sentence
            multilabel: set to process "multilabel" labels array
            sampling_strategy: "unique", "oversampling", or "undersampling"
            num_iterations: if provided explicitly sets the number of pairs to be generated
                where n_pairs = n_iterations * n_sentences * 2 (for pos & neg pairs)
            max_pairs: If not -1, then we only sample pairs until we have certainly reached
                max_pairs pairs.
        """
        super().__init__()
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs = []
        self.neg_pairs = []
        self.sentences = sentences
        self.labels = labels
        self.sentence_labels = list(zip(self.sentences, self.labels))
        self.max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

        if multilabel:
            self.generate_multilabel_pairs()
        else:
            self.generate_pairs()

        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
            self.len_neg_pairs = num_iterations * len(self.sentences)

        elif sampling_strategy == "unique":
            self.len_pos_pairs = len(self.pos_pairs)
            self.len_neg_pairs = len(self.neg_pairs)

        elif sampling_strategy == "undersampling":
            self.len_pos_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = min(len(self.pos_pairs), len(self.neg_pairs))

        elif sampling_strategy == "oversampling":
            self.len_pos_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = max(len(self.pos_pairs), len(self.neg_pairs))

        else:
            raise ValueError("Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'.")

    def generate_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            is_positive = _label == label
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def generate_multilabel_pairs(self) -> None:
        for (_text, _label), (text, label) in shuffle_combinations(self.sentence_labels):
            # logical_and checks if labels are both set for each class
            is_positive = any(np.logical_and(_label, label))
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"sentence_1": _text, "sentence_2": text, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def get_positive_pairs(self) -> List[Dict[str, Union[str, float]]]:
        pairs = []
        for _ in range(self.len_pos_pairs):
            if self.pos_index >= len(self.pos_pairs):
                self.pos_index = 0
            pairs.append(self.pos_pairs[self.pos_index])
            self.pos_index += 1
        return pairs

    def get_negative_pairs(self) -> List[Dict[str, Union[str, float]]]:
        pairs = []
        for _ in range(self.len_neg_pairs):
            if self.neg_index >= len(self.neg_pairs):
                self.neg_index = 0
            pairs.append(self.neg_pairs[self.neg_index])
            self.neg_index += 1
        return pairs

    def __iter__(self):
        for pos_pair, neg_pair in zip_longest(self.get_positive_pairs(), self.get_negative_pairs()):
            if pos_pair is not None:
                yield pos_pair
            if neg_pair is not None:
                yield neg_pair

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs


class HardNegativeContrastiveDataset(ContrastiveDataset):
    def __init__(
        self,
        sentences: List[str],
        labels: List[Union[int, float]],
        multilabel: bool,
        embeddings: np.ndarray,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
        hard_negative_ratio: float = 1.0,
    ) -> None:
        """Generates contrastive pairs with embedding-ranked hard negatives.

        Negative pairs are different-label examples with high cosine similarity
        under the current frozen sentence encoder.
        """
        if multilabel:
            raise ValueError("Hard negative pair mining is only supported for single-label datasets.")
        if not 0.0 <= hard_negative_ratio <= 1.0:
            raise ValueError("hard_negative_ratio must be between 0.0 and 1.0.")

        self.embeddings = self._normalize_embeddings(embeddings)
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_neg_pairs = []
        self.random_neg_pairs = []
        self.hard_neg_index = 0
        self.random_neg_index = 0
        super().__init__(
            sentences,
            labels,
            multilabel=multilabel,
            num_iterations=num_iterations,
            sampling_strategy=sampling_strategy,
            max_pairs=max_pairs,
        )

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    def _make_pair(self, first_index: int, second_index: int, label: float) -> Dict[str, Union[str, float]]:
        return {
            "sentence_1": self.sentences[first_index],
            "sentence_2": self.sentences[second_index],
            "label": label,
        }

    def generate_pairs(self) -> None:
        positive_candidates: List[Tuple[int, int]] = []
        negative_candidates: List[Tuple[float, int, int]] = []
        n_sentences = len(self.sentences)

        for first_index, second_index in np.stack(np.triu_indices(n_sentences, 0), axis=-1):
            if self.labels[first_index] == self.labels[second_index]:
                positive_candidates.append((first_index, second_index))
            else:
                similarity = float(np.dot(self.embeddings[first_index], self.embeddings[second_index]))
                negative_candidates.append((similarity, first_index, second_index))

        rng = np.random.RandomState(seed=42)
        for index in rng.permutation(len(positive_candidates)):
            first_index, second_index = positive_candidates[index]
            self.pos_pairs.append(self._make_pair(first_index, second_index, 1.0))
            if self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg:
                break

        hard_negative_candidates = sorted(negative_candidates, key=lambda item: item[0], reverse=True)
        random_negative_candidates = [negative_candidates[index] for index in rng.permutation(len(negative_candidates))]

        for _, first_index, second_index in hard_negative_candidates:
            self.hard_neg_pairs.append(self._make_pair(first_index, second_index, 0.0))
            if self.max_pos_or_neg != -1 and len(self.hard_neg_pairs) >= self.max_pos_or_neg:
                break

        for _, first_index, second_index in random_negative_candidates:
            self.random_neg_pairs.append(self._make_pair(first_index, second_index, 0.0))
            if self.max_pos_or_neg != -1 and len(self.random_neg_pairs) >= self.max_pos_or_neg:
                break

        self.neg_pairs = self.hard_neg_pairs

    def _cycle_pairs(
        self, pairs: List[Dict[str, Union[str, float]]], index: int, target_length: int
    ) -> Tuple[List[Dict[str, Union[str, float]]], int]:
        cycled_pairs = []
        for _ in range(target_length):
            if index >= len(pairs):
                index = 0
            cycled_pairs.append(pairs[index])
            index += 1
        return cycled_pairs, index

    def get_negative_pairs(self) -> List[Dict[str, Union[str, float]]]:
        hard_length = int(round(self.len_neg_pairs * self.hard_negative_ratio))
        random_length = self.len_neg_pairs - hard_length
        pairs = []

        if hard_length:
            hard_pairs, self.hard_neg_index = self._cycle_pairs(
                self.hard_neg_pairs, self.hard_neg_index, hard_length
            )
            pairs.extend(hard_pairs)
        if random_length:
            random_pairs, self.random_neg_index = self._cycle_pairs(
                self.random_neg_pairs, self.random_neg_index, random_length
            )
            pairs.extend(random_pairs)

        return pairs


class ContrastiveDistillationDataset(ContrastiveDataset):
    def __init__(
        self,
        sentences: List[str],
        cos_sim_matrix: torch.Tensor,
        num_iterations: Optional[None] = None,
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        self.cos_sim_matrix = cos_sim_matrix
        super().__init__(
            sentences,
            [0] * len(sentences),
            multilabel=False,
            num_iterations=num_iterations,
            sampling_strategy=sampling_strategy,
            max_pairs=max_pairs,
        )
        # Internally we store all pairs in pos_pairs, regardless of sampling strategy.
        # After all, without labels, there isn't much of a strategy.
        self.sentence_labels = list(enumerate(self.sentences))

        self.len_neg_pairs = 0
        if num_iterations is not None and num_iterations > 0:
            self.len_pos_pairs = num_iterations * len(self.sentences)
        else:
            self.len_pos_pairs = len(self.pos_pairs)

    def generate_pairs(self) -> None:
        for (text_one, id_one), (text_two, id_two) in shuffle_combinations(self.sentence_labels):
            self.pos_pairs.append(
                {"sentence_1": text_one, "sentence_2": text_two, "label": self.cos_sim_matrix[id_one][id_two]}
            )
            if self.max_pos_or_neg != -1 and len(self.pos_pairs) > self.max_pos_or_neg:
                break
