"""Vectorized regime evaluation kernel."""

import math

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import RegimeParams
from dojiwick.domain.models.value_objects.outcome_models import ConfusionMatrix, RegimeEvaluationReport
from dojiwick.domain.type_aliases import FloatVector, IntVector

from dojiwick.compute.kernels.regime.classify import truth_labels_from_prices


_STATE_LABELS: tuple[int, ...] = (
    MarketState.TRENDING_UP.value,
    MarketState.TRENDING_DOWN.value,
    MarketState.RANGING.value,
    MarketState.VOLATILE.value,
)


def evaluate_regimes(
    *,
    prices: FloatVector,
    predicted_states: IntVector,
    settings: RegimeParams,
    horizons: tuple[int, ...] = (1, 3, 6),
) -> RegimeEvaluationReport:
    """Evaluate predicted states against deterministic forward labels."""

    confusion = np.zeros((len(_STATE_LABELS), len(_STATE_LABELS)), dtype=np.int64)
    label_to_index = {label: index for index, label in enumerate(_STATE_LABELS)}

    total_points = 0
    for horizon in horizons:
        truth = truth_labels_from_prices(prices, horizon, settings)
        valid_len = max(0, len(prices) - horizon)
        if valid_len == 0:
            continue

        actual = truth[:valid_len]
        predicted = predicted_states[:valid_len]
        total_points += valid_len

        for actual_label, predicted_label in zip(actual, predicted, strict=False):
            confusion[label_to_index[int(actual_label)], label_to_index[int(predicted_label)]] += 1

    coarse_confusion = _matrix_to_dict(confusion)
    macro_f1 = _macro_f1(confusion)
    flip_rate, mean_run_length, entropy = _stability_metrics(predicted_states)

    return RegimeEvaluationReport(
        total_points=total_points,
        coarse_confusion=coarse_confusion,
        coarse_macro_f1=macro_f1,
        flip_rate=flip_rate,
        mean_run_length=mean_run_length,
        transition_entropy=entropy,
    )


def _matrix_to_dict(matrix: np.ndarray) -> ConfusionMatrix:
    names = {
        MarketState.TRENDING_UP.value: MarketState.TRENDING_UP.name,
        MarketState.TRENDING_DOWN.value: MarketState.TRENDING_DOWN.name,
        MarketState.RANGING.value: MarketState.RANGING.name,
        MarketState.VOLATILE.value: MarketState.VOLATILE.name,
    }
    result: ConfusionMatrix = {}
    for actual_index, actual_label in enumerate(_STATE_LABELS):
        row: dict[str, int] = {}
        for predicted_index, predicted_label in enumerate(_STATE_LABELS):
            row[names[predicted_label]] = int(matrix[actual_index, predicted_index])
        result[names[actual_label]] = row
    return result


def _macro_f1(matrix: np.ndarray) -> float:
    f1_scores: list[float] = []
    for label_index in range(len(_STATE_LABELS)):
        tp = float(matrix[label_index, label_index])
        fp = float(np.sum(matrix[:, label_index]) - tp)
        fn = float(np.sum(matrix[label_index, :]) - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    return float(np.mean(np.asarray(f1_scores, dtype=np.float64))) if f1_scores else 0.0


def _stability_metrics(states: IntVector) -> tuple[float, float, float]:
    if len(states) <= 1:
        length = float(len(states)) if len(states) else 0.0
        return 0.0, length, 0.0

    flips = np.count_nonzero(states[1:] != states[:-1])
    flip_rate = float(flips / (len(states) - 1))

    runs: list[int] = []
    run = 1
    for index in range(1, len(states)):
        if states[index] == states[index - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    mean_run = float(sum(runs) / len(runs))

    transitions: dict[int, dict[int, int]] = {}
    total = 0
    for index in range(1, len(states)):
        previous = int(states[index - 1])
        current = int(states[index])
        transitions.setdefault(previous, {})
        transitions[previous][current] = transitions[previous].get(current, 0) + 1
        total += 1

    entropy = 0.0
    if total > 0:
        for outgoing in transitions.values():
            count = sum(outgoing.values())
            if count == 0:
                continue
            state_entropy = 0.0
            for value in outgoing.values():
                probability = value / count
                state_entropy -= probability * math.log2(probability)
            entropy += (count / total) * state_entropy

    return flip_rate, mean_run, entropy
