"""
================================================================================================
Time Window Comparative Performance Analysis for Motor Execution EEG Decoding
================================================================================================

Author: Ismail Korkmaz

This script implements a comprehensive comparative performance analysis to evaluate how 
classification performance changes with different time window sizes for feature extraction 
in motor execution EEG decoding using CSP+LDA.

The analysis includes:
- Multiple time window configurations (0.5s to 3.0s)
- Statistical validation using stratified cross-validation
- ANOVA and post-hoc testing for significance
- Comprehensive visualization of results
- Performance metrics and confidence intervals
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import f_oneway, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# MNE and EEG processing
import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score, make_scorer

# Print docstring
print(__doc__)

# Create results directory
results_dir = Path("results/time_window_analysis")
results_dir.mkdir(parents=True, exist_ok=True)


def load_subject_data(subject, task_type='execution'):
    """
    Load and preprocess EEG data for a single subject.
    
    Parameters:
    -----------
    subject : int
        Subject ID number
    task_type : str
        'execution' for motor execution or 'imagery' for motor imagery
    
    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    """
    # Select appropriate runs based on task type
    if task_type == 'execution':
        runs = [5, 9, 13]  # Motor execution: hands vs feet
    else:
        runs = [6, 10, 14]  # Motor imagery: hands vs feet
    
    # Load data
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # Standardize and set montage
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    
    # Rename annotations
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    
    # Set reference and filter
    raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    
    return raw


def extract_epochs_with_window(raw, window_start, window_end, baseline_end=-0.5):
    """
    Extract epochs with specified time window.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    window_start : float
        Start time of the analysis window (in seconds)
    window_end : float
        End time of the analysis window (in seconds)
    baseline_end : float
        End of baseline period (in seconds)
    
    Returns:
    --------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Epoch data
    y : ndarray, shape (n_epochs,)
        Labels
    epochs : mne.Epochs
        Epochs object for visualization
    """
    # Pick EEG channels
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    
    # Create epochs with extended time range to allow different analysis windows
    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=-1.0, tmax=4.0,  # Extended range
        proj=True,
        picks=picks,
        baseline=(None, baseline_end),
        preload=True
    )
    
    # Crop to the desired analysis window
    epochs_windowed = epochs.copy().crop(tmin=window_start, tmax=window_end)
    
    # Extract data and labels
    X = epochs_windowed.get_data()
    y = epochs_windowed.events[:, -1] - 2  # Convert to 0, 1 labels
    
    # Ensure data is in float64 format for compatibility with MNE's CSP
    X = X.astype(np.float64, copy=True)
    y = y.astype(np.int64)
    
    return X, y, epochs


def evaluate_time_window(X, y, cv_folds=5, n_components=4):
    """
    Evaluate CSP+LDA performance for given data.
    
    Parameters:
    -----------
    X : ndarray
        EEG data
    y : ndarray
        Labels
    cv_folds : int
        Number of cross-validation folds
    n_components : int
        Number of CSP components
    
    Returns:
    --------
    scores : dict
        Cross-validation scores
    """
    # Ensure data is properly formatted
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    
    # Validate data dimensions
    if X.ndim != 3:
        raise ValueError(f"Expected 3D data (epochs, channels, times), got {X.ndim}D")
    
    n_epochs, n_channels, n_times = X.shape
    if n_epochs < cv_folds:
        raise ValueError(f"Not enough epochs ({n_epochs}) for {cv_folds}-fold CV")
    
    # Adjust n_components if necessary
    max_components = min(n_channels, n_epochs // cv_folds - 1)
    n_components = min(n_components, max_components)
    
    # Create CSP+LDA pipeline with more robust parameters
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline([('csp', csp), ('lda', lda)])
    
    # Setup cross-validation
    cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
    }
    
    # Perform cross-validation with error handling
    try:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, 
                               return_train_score=False, n_jobs=1, error_score='raise')
    except Exception as e:
        print(f"    Cross-validation failed: {str(e)}")
        # Return dummy scores to continue analysis
        scores = {
            'test_accuracy': np.full(cv_folds, np.nan)
        }
    
    return scores


def run_time_window_analysis(subjects_list=None, task_type='execution', 
                           window_configs=None, cv_folds=5):
    """
    Run comprehensive time window analysis.
    
    Parameters:
    -----------
    subjects_list : list or None
        List of subject IDs to analyze. If None, uses subjects 1-10
    task_type : str
        'execution' or 'imagery'
    window_configs : list or None
        List of (start, end) tuples for time windows
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    results_df : pd.DataFrame
        Results dataframe with all performance metrics
    """
    if subjects_list is None:
        subjects_list = list(range(1, 11))  # Subjects 1-10 for faster analysis
    
    if window_configs is None:
        # Default window configurations
        window_configs = [
            (0.0, 0.5),   # 0.5s window
            (0.0, 1.0),   # 1.0s window  
            (0.5, 1.5),   # 1.0s window (shifted)
            (1.0, 2.0),   # 1.0s window (current standard)
            (0.0, 1.5),   # 1.5s window
            (0.5, 2.0),   # 1.5s window (shifted)
            (0.0, 2.0),   # 2.0s window
            (1.0, 3.0),   # 2.0s window (shifted)
            (0.0, 2.5),   # 2.5s window
            (0.0, 3.0),   # 3.0s window
        ]
    
    results = []
    
    print(f"Starting time window analysis for {len(subjects_list)} subjects...")
    print(f"Window configurations: {len(window_configs)}")
    print(f"Cross-validation: {cv_folds}-fold")
    
    for i, subject in enumerate(subjects_list):
        print(f"\nProcessing Subject {subject} ({i+1}/{len(subjects_list)})...")
        
        try:
            # Load subject data
            raw = load_subject_data(subject, task_type)
            
            for j, (window_start, window_end) in enumerate(window_configs):
                window_duration = window_end - window_start
                window_label = f"{window_start}-{window_end}s"
                
                print(f"  Window {j+1}/{len(window_configs)}: {window_label} (duration: {window_duration}s)")
                
                try:
                    # Extract epochs for this window
                    X, y, epochs = extract_epochs_with_window(raw, window_start, window_end)
                    
                    # Check if we have enough data
                    if len(X) < 20:  # Minimum epochs threshold
                        print(f"    Warning: Only {len(X)} epochs available. Skipping.")
                        continue
                    
                    # Evaluate performance
                    scores = evaluate_time_window(X, y, cv_folds)
                    
                    # Store results (filter out NaN values)
                    valid_scores = scores['test_accuracy'][~np.isnan(scores['test_accuracy'])]
                    
                    if len(valid_scores) > 0:
                        for fold_idx, accuracy in enumerate(scores['test_accuracy']):
                            if not np.isnan(accuracy):
                                results.append({
                                    'subject': subject,
                                    'window_start': window_start,
                                    'window_end': window_end,
                                    'window_duration': window_duration,
                                    'window_label': window_label,
                                    'fold': fold_idx,
                                    'accuracy': accuracy,
                                    'n_epochs': len(X),
                                    'n_channels': X.shape[1],
                                    'n_timepoints': X.shape[2]
                                })
                        
                        print(f"    Accuracy: {np.mean(valid_scores):.3f} ± {np.std(valid_scores):.3f}")
                    else:
                        print(f"    All CV folds failed for this window")
                    
                except Exception as e:
                    print(f"    Error processing window {window_label}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing Subject {subject}: {str(e)}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nWarning: No valid results obtained. This could be due to:")
        print("- Data format issues")
        print("- Insufficient epochs per window")
        print("- Cross-validation failures")
        print("Please check data availability and parameters.")
        return pd.DataFrame()  # Return empty DataFrame instead of raising error
    
    print(f"\nAnalysis completed. Total results: {len(results_df)} records")
    print(f"Successful windows per subject: {len(results_df) // cv_folds}")
    
    return results_df


def perform_statistical_analysis(results_df):
    """
    Perform statistical analysis on the time window results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from time window analysis
    
    Returns:
    --------
    stats_results : dict
        Statistical analysis results
    """
    print("\nPerforming statistical analysis...")
    
    # Calculate summary statistics
    summary_stats = results_df.groupby('window_label')['accuracy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Calculate confidence intervals (95%)
    confidence_intervals = []
    for window_label in results_df['window_label'].unique():
        window_data = results_df[results_df['window_label'] == window_label]['accuracy']
        ci = stats.t.interval(0.95, len(window_data)-1, 
                             loc=np.mean(window_data), 
                             scale=stats.sem(window_data))
        confidence_intervals.append({
            'window_label': window_label,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })
    
    ci_df = pd.DataFrame(confidence_intervals)
    summary_stats = summary_stats.merge(ci_df.set_index('window_label'), left_index=True, right_index=True)
    
    # ANOVA test
    window_groups = [results_df[results_df['window_label'] == label]['accuracy'].values 
                    for label in results_df['window_label'].unique()]
    
    f_stat, p_value_anova = f_oneway(*window_groups)
    
    # Post-hoc analysis (Tukey's HSD)
    tukey_results = pairwise_tukeyhsd(results_df['accuracy'], results_df['window_label'])
    
    # Pairwise t-tests with Bonferroni correction
    window_labels = results_df['window_label'].unique()
    n_comparisons = len(window_labels) * (len(window_labels) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    
    pairwise_results = []
    for i, label1 in enumerate(window_labels):
        for label2 in window_labels[i+1:]:
            data1 = results_df[results_df['window_label'] == label1]['accuracy']
            data2 = results_df[results_df['window_label'] == label2]['accuracy']
            
            t_stat, p_val = ttest_rel(data1, data2)
            
            pairwise_results.append({
                'window1': label1,
                'window2': label2,
                't_statistic': t_stat,
                'p_value': p_val,
                'p_value_bonferroni': p_val * n_comparisons,
                'significant_bonferroni': (p_val * n_comparisons) < 0.05,
                'mean_diff': np.mean(data1) - np.mean(data2)
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    stats_results = {
        'summary_stats': summary_stats,
        'anova_f_stat': f_stat,
        'anova_p_value': p_value_anova,
        'tukey_results': tukey_results,
        'pairwise_tests': pairwise_df,
        'bonferroni_alpha': bonferroni_alpha
    }
    
    return stats_results


def create_visualizations(results_df, stats_results, save_dir):
    """
    Create comprehensive visualizations of the results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    stats_results : dict
        Statistical analysis results
    save_dir : Path
        Directory to save plots
    """
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Box plot with statistical annotations
    plt.figure(figsize=(14, 8))
    
    # Create box plot
    ax = sns.boxplot(data=results_df, x='window_label', y='accuracy')
    plt.title('Classification Accuracy Across Different Time Windows (CSP+LDA)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Window', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add mean points
    means = results_df.groupby('window_label')['accuracy'].mean()
    for i, (window, mean_acc) in enumerate(means.items()):
        plt.plot(i, mean_acc, 'ro', markersize=8, label='Mean' if i == 0 else "")
    
    # Add statistical significance annotations
    y_max = results_df['accuracy'].max()
    y_step = (y_max - results_df['accuracy'].min()) * 0.05
    
    # Find significant pairwise comparisons
    sig_pairs = stats_results['pairwise_tests'][
        stats_results['pairwise_tests']['significant_bonferroni']
    ]
    
    if len(sig_pairs) > 0:
        plt.text(0.02, 0.98, f"* p < {stats_results['bonferroni_alpha']:.4f} (Bonferroni corrected)", 
                transform=ax.transAxes, verticalalignment='top', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'time_window_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean accuracy with confidence intervals
    plt.figure(figsize=(12, 8))
    
    summary_stats = stats_results['summary_stats']
    windows = summary_stats.index
    means = summary_stats['mean']
    ci_lower = summary_stats['ci_lower']
    ci_upper = summary_stats['ci_upper']
    
    # Create bar plot with error bars
    x_pos = np.arange(len(windows))
    bars = plt.bar(x_pos, means, yerr=[means - ci_lower, ci_upper - means], 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    plt.title('Mean Classification Accuracy with 95% Confidence Intervals', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Window', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.xticks(x_pos, windows, rotation=45)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'time_window_means_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance trend by window duration
    plt.figure(figsize=(10, 6))
    
    # Calculate mean accuracy by window duration
    duration_stats = results_df.groupby('window_duration')['accuracy'].agg(['mean', 'std']).reset_index()
    
    plt.errorbar(duration_stats['window_duration'], duration_stats['mean'], 
                yerr=duration_stats['std'], marker='o', linewidth=2, markersize=8)
    plt.title('Classification Accuracy vs. Time Window Duration', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Window Duration (seconds)', fontsize=14)
    plt.ylabel('Mean Classification Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_vs_duration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Subject-wise performance heatmap
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    subject_window_means = results_df.groupby(['subject', 'window_label'])['accuracy'].mean().unstack()
    
    sns.heatmap(subject_window_means, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Classification Accuracy'})
    plt.title('Subject-wise Classification Accuracy Across Time Windows', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Window', fontsize=14)
    plt.ylabel('Subject ID', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'subject_window_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {save_dir}")


def save_results(results_df, stats_results, save_dir):
    """
    Save all results to files.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    stats_results : dict
        Statistical analysis results
    save_dir : Path
        Directory to save results
    """
    print(f"\nSaving results to {save_dir}...")
    
    # Save raw results
    results_df.to_csv(save_dir / 'time_window_results.csv', index=False)
    
    # Save summary statistics
    stats_results['summary_stats'].to_csv(save_dir / 'summary_statistics.csv')
    
    # Save pairwise comparisons
    stats_results['pairwise_tests'].to_csv(save_dir / 'pairwise_comparisons.csv', index=False)
    
    # Save ANOVA results and other statistics to text file
    with open(save_dir / 'statistical_analysis.txt', 'w') as f:
        f.write("Time Window Comparative Analysis - Statistical Results\n")
        f.write("=" * 55 + "\n\n")
        
        f.write("ANOVA Results:\n")
        f.write(f"F-statistic: {stats_results['anova_f_stat']:.4f}\n")
        f.write(f"p-value: {stats_results['anova_p_value']:.6f}\n")
        f.write(f"Significant: {'Yes' if stats_results['anova_p_value'] < 0.05 else 'No'}\n\n")
        
        f.write("Bonferroni Correction:\n")
        f.write(f"Corrected alpha level: {stats_results['bonferroni_alpha']:.6f}\n\n")
        
        f.write("Tukey's HSD Results:\n")
        f.write(str(stats_results['tukey_results']))
        f.write("\n\nSignificant Pairwise Comparisons (Bonferroni corrected):\n")
        
        sig_comparisons = stats_results['pairwise_tests'][
            stats_results['pairwise_tests']['significant_bonferroni']
        ]
        
        if len(sig_comparisons) > 0:
            for _, row in sig_comparisons.iterrows():
                f.write(f"{row['window1']} vs {row['window2']}: ")
                f.write(f"mean difference = {row['mean_diff']:.4f}, ")
                f.write(f"p = {row['p_value_bonferroni']:.6f}\n")
        else:
            f.write("No significant pairwise differences found.\n")
    
    print("Results saved successfully!")


def main():
    """
    Main function to run the complete time window analysis.
    """
    print("Time Window Comparative Performance Analysis")
    print("=" * 50)
    
    # Configuration
    subjects_list = list(range(1, 110))  # Start with subjects 1-5 for testing
    task_type = 'execution'  # Motor execution task
    cv_folds = 10
    
    # Define time windows to compare
    window_configs = [
        (0.0, 1.0),
        (0.5, 1.5),
        (1.0, 2.0),
        (2.0, 3.0),
    ]
    
    try:
        # Run analysis
        results_df = run_time_window_analysis(
            subjects_list=subjects_list,
            task_type=task_type,
            window_configs=window_configs,
            cv_folds=cv_folds
        )
        
        # Check if we have valid results
        if len(results_df) == 0:
            print("\nNo valid results obtained. Analysis terminated.")
            return
        
        # Perform statistical analysis
        stats_results = perform_statistical_analysis(results_df)
        
        # Create visualizations
        create_visualizations(results_df, stats_results, results_dir)
        
        # Save results
        save_results(results_df, stats_results, results_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        
        print(f"\nBest performing window: {stats_results['summary_stats']['mean'].idxmax()}")
        print(f"Best accuracy: {stats_results['summary_stats']['mean'].max():.4f}")
        
        print(f"\nANOVA result: F = {stats_results['anova_f_stat']:.4f}, p = {stats_results['anova_p_value']:.6f}")
        print(f"Significant differences: {'Yes' if stats_results['anova_p_value'] < 0.05 else 'No'}")
        
        n_sig_pairs = len(stats_results['pairwise_tests'][
            stats_results['pairwise_tests']['significant_bonferroni']
        ])
        print(f"Significant pairwise comparisons: {n_sig_pairs}")
        
        print(f"\nResults saved to: {results_dir}")
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main() 