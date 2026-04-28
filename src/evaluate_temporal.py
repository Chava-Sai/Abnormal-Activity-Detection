"""
Temporal IoU-based Evaluation for Violence Detection.

Computes mAP@tIoU metrics by:
1. Generating temporal proposals from model predictions
2. Matching to ground truth temporal boundaries
3. Computing AP at multiple IoU thresholds
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import UCFCrimeDataset
from temporal_metrics import (
    TemporalEvaluator, 
    generate_proposals_from_scores,
)


def normalize_video_id(video_name: str) -> str:
    """Normalize video IDs so GT keys match feature file names."""
    name = video_name
    if name.endswith('.npy'):
        name = name[:-4]
    if name.endswith('_i3d'):
        name = name[:-4]
    return name


@torch.no_grad()
def extract_predictions_from_model(
    model: ViolenceDetector,
    dataset,
    device: torch.device,
    score_threshold: float = 0.5,
    frames_per_segment: int = 16,
    use_boundary_scores: bool = False,
) -> Dict[str, Dict]:
   
    model.eval()
    predictions_dict = {}
    
    for i in range(len(dataset)):
        item = dataset[i]
        feat = item['features'].unsqueeze(0).to(device)  # (1, T, 2048)
        label = item['label'].item()
        cat_id = item['cat_id'].item()
        fpath = item['filepath']
        
        # Get video ID from filename
        video_id = normalize_video_id(Path(fpath).stem)
        
        # Run inference
        out = model(feat)
        segment_scores = out['trn_scores'].squeeze(0).cpu().numpy()  # (T,)
        
        # Get boundary scores if available
        boundary_scores = None
        if use_boundary_scores and 'boundary' in out:
            boundary_scores = out['boundary'].squeeze(0).cpu().numpy()  # (T-1,)
        
        # Generate proposals
        proposals = generate_proposals_from_scores(
            segment_scores,
            boundary_scores=boundary_scores,
            score_threshold=score_threshold,
            merge_threshold=0,  # No merging for strict evaluation
        )
        
        # Get category name
        cat_id_to_name = {
            0: 'Normal', 1: 'Abuse', 2: 'Fighting',
            3: 'Shooting', 4: 'Explosion', 5: 'Robbery', 6: 'Riot',
        }
        category = cat_id_to_name.get(cat_id, f'cat_{cat_id}')
        
        predictions_dict[video_id] = {
            'proposals': proposals,
            'segment_scores': segment_scores,
            'boundary_scores': boundary_scores,
            'is_anomalous': bool(label),
            'category': category,
        }
    
    return predictions_dict


def load_ground_truth_from_json(
    gt_file: str,
    frames_per_segment: int = 16,
) -> Dict[str, List[Dict]]:

    with open(gt_file, 'r') as f:
        gt_json = json.load(f)
    
    gt_dict = {}
    for video_name, annotations in gt_json.items():
        normalized_name = normalize_video_id(video_name)
        gt_segs = []
        for ann in annotations:
            # Convert frames to segments
            seg_start = ann['start_frame'] // frames_per_segment
            seg_end = ann['end_frame'] // frames_per_segment
            gt_segs.append({
                'start': seg_start,
                'end': seg_end,
            })
        gt_dict[normalized_name] = gt_segs
    
    return gt_dict


def evaluate_checkpoint_temporal(
    checkpoint_path: str,
    feature_dir: str,
    list_file: str,
    gt_file: str,
    device: torch.device,
    score_threshold: float = 0.5,
    frames_per_segment: int = 16,
    num_segments: int = 32,
    use_boundary_scores: bool = False,
    iou_thresholds: List[float] = None,
) -> Dict:
    """
    Load checkpoint and compute mAP@tIoU.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.75]
    
    print(f"\n{'='*70}")
    print(f"  Temporal Localization Evaluation (mAP@tIoU)")
    print(f"{'='*70}")
    
    # Load model
    print(f"\n[1/3] Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('args', {})
    
    model = ViolenceDetector(
        input_dim=2048,
        num_classes=7,
        d_model=model_args.get('d_model', 512),
        nhead=model_args.get('nhead', 8),
        trn_layers=model_args.get('trn_layers', 2),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    epoch = ckpt.get('epoch', '?')
    print(f"  Loaded checkpoint from epoch {epoch}")
    
    # Load dataset
    print(f"\n[2/3] Loading dataset...")
    dataset = UCFCrimeDataset(
        feature_dir, list_file,
        mode='test', num_segments=num_segments, violence_only=False
    )
    print(f"  Dataset size: {len(dataset)} videos")
    
    # Generate predictions
    print(f"\n[3/3] Running inference and generating proposals...")
    predictions_dict = extract_predictions_from_model(
        model, dataset, device,
        score_threshold=score_threshold,
        frames_per_segment=frames_per_segment,
        use_boundary_scores=use_boundary_scores,
    )
    
    # Load ground truth
    print(f"\n[4/4] Loading ground truth annotations from {gt_file}...")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"GT file not found: {gt_file}")
    
    gt_dict = load_ground_truth_from_json(gt_file, frames_per_segment)
    print(f"  Loaded GT for {len(gt_dict)} videos")
    
    # Evaluate
    print(f"\nComputing mAP@tIoU (thresholds: {iou_thresholds})...")
    evaluator = TemporalEvaluator(iou_thresholds=iou_thresholds)
    
    matched_videos = 0
    for video_id, pred_data in predictions_dict.items():
        if video_id not in gt_dict:
            continue
        
        predictions = pred_data['proposals']
        ground_truth = gt_dict[video_id]
        
        evaluator.add_predictions(video_id, predictions, ground_truth)
        matched_videos += 1
    
    print(f"  Matched {matched_videos} videos with GT annotations")
    
    # Get results
    metrics = {}
    evaluator.print_results()
    metrics['temporal'] = evaluator.compute_metrics()
    
    # Add statistics
    metrics['stats'] = {
        'checkpoint_epoch': epoch,
        'score_threshold': score_threshold,
        'use_boundary_scores': use_boundary_scores,
        'iou_thresholds': iou_thresholds,
        'matched_videos': matched_videos,
    }
    
    return metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    metrics = evaluate_checkpoint_temporal(
        checkpoint_path=args.checkpoint,
        feature_dir=os.path.expanduser(args.feature_dir),
        list_file=os.path.expanduser(args.list_file),
        gt_file=os.path.expanduser(args.gt_file),
        device=device,
        score_threshold=args.score_threshold,
        frames_per_segment=args.frames_per_segment,
        num_segments=args.num_segments,
        use_boundary_scores=args.use_boundary,
        iou_thresholds=args.iou_thresholds,
    )
    
    if args.output_json:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        metrics_serializable = convert_to_serializable(metrics)
        with open(args.output_json, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate violence detection with temporal IoU metrics'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--feature_dir', required=True,
                        help='Path to I3D feature directory')
    parser.add_argument('--list_file', required=True,
                        help='List file (test split)')
    parser.add_argument('--gt_file', required=True,
                        help='JSON file with ground truth temporal annotations')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Anomaly score threshold for proposal generation')
    parser.add_argument('--frames_per_segment', type=int, default=16,
                        help='Frames per segment')
    parser.add_argument('--num_segments', type=int, default=32,
                        help='Number of segments per video (fixed)')
    parser.add_argument('--use_boundary', action='store_true',
                        help='Use boundary head for proposal refinement')
    parser.add_argument('--iou_thresholds', type=float, nargs='+',
                        default=[0.3, 0.5, 0.75],
                        help='IoU thresholds for AP computation')
    parser.add_argument('--output_json', default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    main(args)
