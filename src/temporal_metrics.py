"""
Temporal Intersection over Union (tIoU) based Average Precision metrics.

For weakly-supervised video anomaly detection with temporal localization.
Computes mAP@tIoU at multiple IoU thresholds (0.3, 0.5, 0.75).

Prediction format:
    predicted_proposals: list of dicts with keys:
        - 'start':     int (segment index)
        - 'end':       int (segment index, inclusive)
        - 'score':     float (anomaly score [0, 1])
        - 'segments':  list of segment scores (optional, for debugging)

Ground truth format:
    gt_boundaries: list of dicts with keys:
        - 'start':     int (segment index)
        - 'end':       int (segment index, inclusive)
        - 'label':     int (category ID, optional)
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def compute_temporal_iou(pred_start: int, pred_end: int, 
                        gt_start: int, gt_end: int) -> float:
    """
    Compute temporal Intersection over Union between prediction and GT.
    
    Args:
        pred_start, pred_end: segment indices of prediction (inclusive)
        gt_start, gt_end:     segment indices of ground truth (inclusive)
    
    Returns:
        tIoU: float in [0, 1]
    """
    # Convert to inclusive ranges [start, end]
    pred_len = pred_end - pred_start + 1
    gt_len = gt_end - gt_start + 1
    
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    
    if inter_start > inter_end:
        return 0.0
    
    inter_len = inter_end - inter_start + 1
    
    # Union
    union_len = pred_len + gt_len - inter_len
    
    iou = inter_len / union_len if union_len > 0 else 0.0
    return iou


def generate_proposals_from_scores(segment_scores: np.ndarray,
                                   boundary_scores: Optional[np.ndarray] = None,
                                   score_threshold: float = 0.5,
                                   merge_threshold: int = 1) -> List[Dict]:
    """
    Generate temporal proposals from segment-level anomaly scores.
    
    Detects contiguous regions where score > score_threshold,
    optionally using boundary confidence to refine boundaries.
    
    Args:
        segment_scores: (T,) array of segment-level scores
        boundary_scores: (T-1,) array of boundary transition confidence (optional)
        score_threshold: score threshold for event detection
        merge_threshold: merge segments separated by fewer than this many frames
    
    Returns:
        proposals: list of dicts with keys 'start', 'end', 'score'
    """
    T = len(segment_scores)
    proposals = []
    
    # Find segments above threshold
    above_thresh = segment_scores > score_threshold
    
    # Detect contiguous regions
    boundaries = np.diff(above_thresh.astype(int))
    starts = np.where(boundaries == 1)[0] + 1
    ends = np.where(boundaries == -1)[0]
    
    # Handle edge cases
    if above_thresh[0]:
        starts = np.concatenate([[0], starts])
    if above_thresh[-1]:
        ends = np.concatenate([ends, [T-1]])
    
    # Create proposals
    for start, end in zip(starts, ends):
        if end >= start:  # valid proposal
            # Use max score within proposal
            prop_score = float(segment_scores[start:end+1].max())
            proposals.append({
                'start': int(start),
                'end': int(end),
                'score': prop_score,
            })
    
    # Merge nearby proposals (optional)
    if merge_threshold > 0:
        proposals = merge_nearby_proposals(proposals, merge_threshold)
    
    return proposals


def merge_nearby_proposals(proposals: List[Dict], merge_threshold: int) -> List[Dict]:
    """
    Merge proposals separated by fewer than merge_threshold segments.
    """
    if len(proposals) <= 1:
        return proposals
    
    merged = []
    current = proposals[0].copy()
    
    for next_prop in proposals[1:]:
        gap = next_prop['start'] - current['end'] - 1
        
        if gap < merge_threshold:
            # Merge: extend current proposal
            current['end'] = next_prop['end']
            current['score'] = max(current['score'], next_prop['score'])
        else:
            # Gap too large, save current and start new
            merged.append(current)
            current = next_prop.copy()
    
    merged.append(current)
    return merged


def match_proposals_to_gt(predictions: List[Dict],
                          ground_truths: List[Dict],
                          iou_threshold: float = 0.5) -> Tuple[List[bool], List[bool]]:
    """
    Match predicted proposals to ground truth annotations at a given IoU threshold.
    
    Uses greedy matching: for each prediction (sorted by score desc),
    assign it to the highest-IoU GT that hasn't been matched yet.
    
    Args:
        predictions: list of dicts with 'start', 'end', 'score'
        ground_truths: list of dicts with 'start', 'end'
        iou_threshold: only match if IoU >= this threshold
    
    Returns:
        (tp, conf): arrays for computing AP
            tp[i] = 1 if prediction i matched to GT with IoU >= threshold
            conf[i] = score of prediction i
    """
    if len(predictions) == 0:
        return np.array([]), np.array([])
    
    if len(ground_truths) == 0:
        # All predictions are false positives
        return np.zeros(len(predictions)), np.array([p['score'] for p in predictions])
    
    # Sort predictions by score (descending)
    sorted_preds = sorted(enumerate(predictions), key=lambda x: -x[1]['score'])
    
    # Track which GTs have been matched
    gt_matched = [False] * len(ground_truths)
    tp = np.zeros(len(predictions))
    conf = np.array([p['score'] for p in predictions])
    
    # Greedy matching
    for pred_idx, pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            
            iou = compute_temporal_iou(
                pred['start'], pred['end'],
                gt['start'], gt['end']
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Match if IoU meets threshold
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
    
    return tp, conf


def compute_ap_at_iou(tp: np.ndarray, conf: np.ndarray) -> float:
    """
    Compute Average Precision from TP/FP labels and confidence scores.
    
    Args:
        tp: binary array (1 = true positive, 0 = false positive)
        conf: confidence scores (same length as tp)
    
    Returns:
        ap: float in [0, 1]
    """
    if len(tp) == 0:
        return 0.0
    
    # Sort by confidence (descending)
    sorted_idx = np.argsort(-conf)
    tp_sorted = tp[sorted_idx]
    
    # Cumulative TP and total positives
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)
    
    # Precision and recall
    recall = tp_cumsum / len(np.where(tp == 1)[0]) if np.any(tp == 1) else np.zeros_like(tp_cumsum)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP (area under P-R curve)
    # Use all-points interpolation
    ap = 0.0
    for i in range(len(precision)):
        if i == 0 or recall[i] != recall[i-1]:
            p = precision[i:].max() if i < len(precision) else 0
            ap += p * (recall[i] - (recall[i-1] if i > 0 else 0))
    
    return float(ap)


class TemporalEvaluator:
    """
    Compute mAP@tIoU for temporal localization.
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Args:
            iou_thresholds: list of IoU thresholds for AP computation
                           (default: [0.3, 0.5, 0.75])
        """
        if iou_thresholds is None:
            iou_thresholds = [0.3, 0.5, 0.75]
        
        self.iou_thresholds = sorted(iou_thresholds)
        self.predictions_by_video = defaultdict(list)
        self.gts_by_video = defaultdict(list)
    
    def add_predictions(self, video_id: str, 
                       predictions: List[Dict],
                       ground_truths: List[Dict]):
        """
        Add predictions and ground truth for a video.
        
        Args:
            video_id: unique video identifier
            predictions: list of dicts with 'start', 'end', 'score'
            ground_truths: list of dicts with 'start', 'end'
        """
        self.predictions_by_video[video_id] = predictions
        self.gts_by_video[video_id] = ground_truths
    
    def compute_metrics(self) -> Dict:
        """
        Compute mAP at all IoU thresholds.
        
        Returns:
            metrics: dict with keys:
                - 'mAP@{iou}': AP at this IoU threshold
                - 'mAP': average across all thresholds
                - 'breakdown': per-video AP values
        """
        metrics = {}
        all_ap_values = defaultdict(list)
        
        for iou_threshold in self.iou_thresholds:
            ap_sum = 0.0
            n_videos_with_gt = 0
            video_aps = {}
            
            for video_id in self.gts_by_video.keys():
                gts = self.gts_by_video[video_id]
                preds = self.predictions_by_video.get(video_id, [])
                
                if len(gts) == 0:
                    continue  # Skip videos without GT
                
                # Match predictions to GT
                tp, conf = match_proposals_to_gt(preds, gts, iou_threshold)
                
                # Compute AP for this video
                if len(tp) > 0:
                    ap = compute_ap_at_iou(tp, conf)
                else:
                    ap = 0.0  # No predictions
                
                video_aps[video_id] = ap
                ap_sum += ap
                n_videos_with_gt += 1
                all_ap_values[iou_threshold].append(ap)
            
            if n_videos_with_gt > 0:
                map_score = ap_sum / n_videos_with_gt
            else:
                map_score = 0.0
            
            key = f'mAP@{iou_threshold:.2f}'
            metrics[key] = map_score
            metrics[f'{key}_breakdown'] = video_aps
        
        # Overall mAP (average across all thresholds)
        all_maps = [metrics[f'mAP@{iou:.2f}'] for iou in self.iou_thresholds]
        metrics['mAP'] = float(np.mean(all_maps))
        
        return metrics
    
    def print_results(self):
        """Pretty print evaluation results."""
        metrics = self.compute_metrics()
        
        print(f"\n{'='*60}")
        print(f"  Temporal Localization (tIoU-based) Results")
        print(f"{'='*60}")
        
        for iou in self.iou_thresholds:
            key = f'mAP@{iou:.2f}'
            if key in metrics:
                ap = metrics[key]
                bar = '█' * int(ap * 30)
                print(f"  {key:15s}: {ap:.4f}  {bar}")
        
        print(f"  {'─'*60}")
        print(f"  Overall mAP    : {metrics['mAP']:.4f}")
        print(f"{'='*60}\n")
        
        return metrics


# Example / Test
if __name__ == '__main__':
    # Create dummy example
    evaluator = TemporalEvaluator([0.3, 0.5, 0.75])
    
    # Video 1: Perfect prediction
    pred_scores = np.array([0.1, 0.9, 0.95, 0.8, 0.1, 0.1])
    props = generate_proposals_from_scores(pred_scores, score_threshold=0.5)
    gts = [{'start': 1, 'end': 3}]
    evaluator.add_predictions('video_1', props, gts)
    
    # Video 2: Partial overlap
    pred_scores2 = np.array([0.1, 0.1, 0.85, 0.9, 0.7, 0.2])
    props2 = generate_proposals_from_scores(pred_scores2, score_threshold=0.5)
    gts2 = [{'start': 1, 'end': 4}]
    evaluator.add_predictions('video_2', props2, gts2)
    
    # Compute and print
    evaluator.print_results()
