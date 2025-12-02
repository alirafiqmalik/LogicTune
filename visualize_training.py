"""
Visualization Script for Training Results

Generates all required plots:
1. DPO Losses, Accuracies, Marginal Preferences vs Epoch (mean over 5 seeds, min-max shading)
2. Box and Whisker Plot for specifications satisfied vs epoch
3. Grouped Bar Chart for per-specification satisfaction rates

Usage:
    python visualize_training.py --metrics training_metrics.json --eval evaluation_results.json
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from logictune import TrainingVisualizer, generate_sample_data_for_testing


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LogicTune training results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to training metrics JSON file"
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate demo plots with sample data"
    )
    
    args = parser.parse_args()
    
    visualizer = TrainingVisualizer(output_dir=args.output_dir)
    
    if args.demo:
        print("="*70)
        print(" GENERATING DEMO VISUALIZATIONS")
        print("="*70)
        
        training_metrics, epoch_scores, spec_satisfaction = generate_sample_data_for_testing()
        
        # Get number of epochs from the shape of metrics (2D array: num_seeds x num_epochs)
        num_epochs = len(training_metrics['loss'][0]) if isinstance(training_metrics['loss'][0], list) else len(training_metrics['loss'])
        epochs = list(range(num_epochs))
        
        print("\n1. Plotting training metrics (DPO Losses, Accuracies, Marginal Preferences)...")
        visualizer.plot_training_metrics(
            metrics=training_metrics,
            epochs=epochs,
            num_seeds=5
        )
        
        print("\n2. Plotting specification satisfaction by epoch...")
        visualizer.plot_specification_satisfaction_by_epoch(
            epoch_scores=epoch_scores
        )
        
        print("\n3. Plotting per-specification satisfaction rates...")
        visualizer.plot_per_specification_satisfaction(
            spec_satisfaction=spec_satisfaction
        )
        
        print(f"\n{'='*70}")
        print(f" ✓ Demo visualizations complete!")
        print(f" ✓ Saved to: {args.output_dir}")
        print(f"{'='*70}\n")
        
    elif args.metrics:
        visualizer.create_all_plots_from_training_log(
            log_file=args.metrics,
            evaluation_file=args.eval
        )
    else:
        print("Error: Must provide --metrics or use --demo flag")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

