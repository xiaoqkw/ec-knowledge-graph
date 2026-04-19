import evaluate
from transformers import EvalPrediction

from configuration.config import ENTITY_TYPES


def build_metrics(id_to_label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(prediction: EvalPrediction):
        logits = prediction.predictions
        label_ids = prediction.label_ids
        predictions = logits.argmax(axis=-1)

        all_predictions = []
        all_labels = []
        for pred_ids, gold_ids in zip(predictions, label_ids):
            valid_mask = gold_ids != -100
            pred_sequence = [id_to_label[int(item)] for item in pred_ids[valid_mask]]
            gold_sequence = [id_to_label[int(item)] for item in gold_ids[valid_mask]]
            all_predictions.append(pred_sequence)
            all_labels.append(gold_sequence)

        raw_metrics = seqeval.compute(
            predictions=all_predictions,
            references=all_labels,
        )

        metrics = {
            "overall_precision": raw_metrics["overall_precision"],
            "overall_recall": raw_metrics["overall_recall"],
            "overall_f1": raw_metrics["overall_f1"],
            "overall_accuracy": raw_metrics["overall_accuracy"],
        }

        for entity_type in ENTITY_TYPES:
            entity_metrics = raw_metrics.get(entity_type, {})
            metrics[f"{entity_type.lower()}_precision"] = entity_metrics.get("precision", 0.0)
            metrics[f"{entity_type.lower()}_recall"] = entity_metrics.get("recall", 0.0)
            metrics[f"{entity_type.lower()}_f1"] = entity_metrics.get("f1", 0.0)

        return metrics

    return compute_metrics
