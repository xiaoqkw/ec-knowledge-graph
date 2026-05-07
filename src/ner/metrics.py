import evaluate
from transformers import EvalPrediction

from configuration.config import ENTITY_TYPES


SEQEVAL = evaluate.load("seqeval")


def to_type_agnostic_label(label: str) -> str:
    if label == "O":
        return label
    prefix, _ = label.split("-", maxsplit=1)
    return f"{prefix}-TAG"


def collect_sequences(predictions, label_ids, id_to_label):
    all_predictions = []
    all_labels = []
    span_only_predictions = []
    span_only_labels = []
    for pred_ids, gold_ids in zip(predictions, label_ids):
        valid_mask = gold_ids != -100
        pred_sequence = [id_to_label[int(item)] for item in pred_ids[valid_mask]]
        gold_sequence = [id_to_label[int(item)] for item in gold_ids[valid_mask]]
        all_predictions.append(pred_sequence)
        all_labels.append(gold_sequence)
        span_only_predictions.append(
            [to_type_agnostic_label(label) for label in pred_sequence]
        )
        span_only_labels.append(
            [to_type_agnostic_label(label) for label in gold_sequence]
        )
    return all_predictions, all_labels, span_only_predictions, span_only_labels


def compute_metrics_from_sequences(all_predictions, all_labels, span_only_predictions, span_only_labels):
    raw_metrics = SEQEVAL.compute(
        predictions=all_predictions,
        references=all_labels,
    )
    span_only_metrics = SEQEVAL.compute(
        predictions=span_only_predictions,
        references=span_only_labels,
    )

    metrics = {
        "overall_precision": raw_metrics["overall_precision"],
        "overall_recall": raw_metrics["overall_recall"],
        "overall_f1": raw_metrics["overall_f1"],
        "overall_accuracy": raw_metrics["overall_accuracy"],
        "type_agnostic_span_precision": span_only_metrics["overall_precision"],
        "type_agnostic_span_recall": span_only_metrics["overall_recall"],
        "type_agnostic_span_f1": span_only_metrics["overall_f1"],
    }

    for entity_type in ENTITY_TYPES:
        entity_metrics = raw_metrics.get(entity_type, {})
        metrics[f"{entity_type.lower()}_precision"] = entity_metrics.get("precision", 0.0)
        metrics[f"{entity_type.lower()}_recall"] = entity_metrics.get("recall", 0.0)
        metrics[f"{entity_type.lower()}_f1"] = entity_metrics.get("f1", 0.0)

    return metrics


def compute_metrics_from_predictions(predictions, label_ids, id_to_label):
    sequences = collect_sequences(predictions, label_ids, id_to_label)
    return compute_metrics_from_sequences(*sequences)


def build_metrics(id_to_label):
    def compute_metrics(prediction: EvalPrediction):
        logits = prediction.predictions
        label_ids = prediction.label_ids
        predictions = logits.argmax(axis=-1)
        return compute_metrics_from_predictions(predictions, label_ids, id_to_label)

    return compute_metrics
