"""
================================================================================================
Motor Imagery/Execution Decoding from EEG data using Various Feature Extraction and Classification Methods
with Real-time Performance Assessment
================================================================================================

Author: Ismail Korkmaz

This script is a unified pipeline that combines multiple feature extraction and classification methods:
- CSP (Common Spatial Patterns) with KNN, SVM, LDA, and MLP classifiers
- Statistical Features with KNN, SVM, LDA, and MLP classifiers

It processes EEG data for both motor imagery and motor execution tasks, evaluates performance
using cross-validation, and includes real-time performance assessment for BCI feasibility.
"""

# Importing necessary libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import timeit
import psutil
import gc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# MNE libraries
import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

# Print docstring
print(__doc__)


def measure_inference_performance(clf, X_test, y_test, n_single_tests=100):
    """
    Measure inference performance metrics for real-time BCI assessment.
    
    Parameters:
    -----------
    clf : sklearn estimator
        Trained classifier
    X_test : array
        Test data
    y_test : array
        Test labels
    n_single_tests : int
        Number of single predictions to test for timing
    
    Returns:
    --------
    performance_metrics : dict
        Dictionary containing timing and resource usage metrics
    """
    performance_metrics = {}
    
    # Measure single prediction time
    single_times = []
    for i in range(min(n_single_tests, len(X_test))):
        start_time = time.perf_counter()
        _ = clf.predict(X_test[i:i+1])
        end_time = time.perf_counter()
        single_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    performance_metrics['single_prediction_time_ms'] = {
        'mean': np.mean(single_times),
        'std': np.std(single_times),
        'min': np.min(single_times),
        'max': np.max(single_times)
    }
    
    # Measure batch prediction time
    batch_start = time.perf_counter()
    _ = clf.predict(X_test)
    batch_end = time.perf_counter()
    batch_time_ms = (batch_end - batch_start) * 1000
    performance_metrics['batch_prediction_time_ms'] = batch_time_ms
    performance_metrics['batch_time_per_sample_ms'] = batch_time_ms / len(X_test)
    
    # Measure CPU and memory usage during batch prediction
    process = psutil.Process()
    
    # Memory usage before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # CPU and memory usage during prediction
    cpu_percent_before = process.cpu_percent()
    start_time = time.time()
    
    # Run prediction with resource monitoring
    _ = clf.predict(X_test)
    
    end_time = time.time()
    cpu_percent_after = process.cpu_percent()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    performance_metrics['cpu_usage_percent'] = cpu_percent_after
    performance_metrics['memory_usage_mb'] = memory_after
    performance_metrics['memory_increase_mb'] = memory_after - memory_before
    
    # Assess real-time feasibility (assuming 250Hz sampling rate, 1-second windows)
    # For real-time BCI, we typically need predictions faster than the window update rate
    real_time_threshold_ms = 100  # Conservative threshold for real-time processing
    performance_metrics['real_time_feasible'] = performance_metrics['single_prediction_time_ms']['mean'] < real_time_threshold_ms
    performance_metrics['real_time_threshold_ms'] = real_time_threshold_ms
    
    # Calculate maximum possible sampling rate based on inference time
    max_predictions_per_second = 1000 / performance_metrics['single_prediction_time_ms']['mean']
    performance_metrics['max_predictions_per_second'] = max_predictions_per_second
    
    return performance_metrics


def test_model_complexity(clf):
    """
    Assess model complexity and memory footprint.
    
    Parameters:
    -----------
    clf : sklearn estimator
        Trained classifier
    
    Returns:
    --------
    complexity_metrics : dict
        Dictionary containing model complexity metrics
    """
    complexity_metrics = {}
    
    # Try to get model parameters count
    total_params = 0
    try:
        if hasattr(clf, 'named_steps'):
            # Pipeline case
            for step_name, step in clf.named_steps.items():
                if hasattr(step, 'coef_'):
                    if step.coef_ is not None:
                        total_params += np.prod(step.coef_.shape)
                if hasattr(step, 'intercept_'):
                    if step.intercept_ is not None:
                        total_params += np.prod(step.intercept_.shape)
        else:
            # Single estimator case
            if hasattr(clf, 'coef_'):
                if clf.coef_ is not None:
                    total_params += np.prod(clf.coef_.shape)
            if hasattr(clf, 'intercept_'):
                if clf.intercept_ is not None:
                    total_params += np.prod(clf.intercept_.shape)
    except:
        total_params = 0
    
    complexity_metrics['total_parameters'] = total_params
    
    # Model memory footprint (rough estimate)
    try:
        import pickle
        model_bytes = len(pickle.dumps(clf))
        complexity_metrics['model_size_kb'] = model_bytes / 1024
    except:
        complexity_metrics['model_size_kb'] = 0
    
    return complexity_metrics


def process_subject(subject, feature_method='csp', classifier='knn', task_type='execution'):
    """
    Process a single subject's EEG data with specified feature extraction and classification methods.
    
    Parameters:
    -----------
    subject : int
        Subject ID number
    feature_method : str
        Feature extraction method ('csp' or 'stats')
    classifier : str
        Classifier type ('knn', 'svm', 'lda', or 'mlp')
    task_type : str
        'execution' for motor execution or 'imagery' for motor imagery
    
    Returns:
    --------
    subject_results : dict
        Dictionary containing performance scores, timing metrics, and resource usage
    """
    # Set parameters for data processing
    tmin, tmax = -1.0, 4.0
    
    # Select appropriate runs based on task type
    if task_type == 'execution':
        runs = [5, 9, 13]  # motor execution: hands vs feet
    else:  # imagery
        runs = [6, 10, 14]  # motor imagery: hands vs feet
    
    # Load and preprocess data
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    raw.set_eeg_reference(projection=True)
    
    # Apply band-pass filter
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    
    # Pick EEG channels
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    
    # Extract epochs
    epochs = Epochs(
        raw,
        event_id=["hands", "feet"],
        tmin=tmin, tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True
    )
    
    # Extract data and labels
    X = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data()
    y = epochs.events[:, -1] - 2  # Convert event codes to 0/1
    
    # Feature extraction
    if feature_method == 'csp':
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        
        # Set up the appropriate classifier
        if classifier == 'knn':
            clf_obj = KNeighborsClassifier(n_neighbors=5)
            clf = Pipeline([('csp', csp), ('knn', clf_obj)])
        elif classifier == 'svm':
            clf_obj = SVC(kernel='linear')
            clf = Pipeline([('csp', csp), ('svm', clf_obj)])
        elif classifier == 'lda':
            clf_obj = LinearDiscriminantAnalysis()
            clf = Pipeline([('csp', csp), ('lda', clf_obj)])
        elif classifier == 'mlp':
            clf_obj = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
            clf = Pipeline([('csp', csp), ('mlp', clf_obj)])
        
        # Evaluate CSP patterns
        result_dir = f'./results/{feature_method}_{classifier}'
        os.makedirs(result_dir, exist_ok=True)
        
        # After processing, plot CSP patterns
        csp.fit_transform(X, y)
        csp.plot_patterns(epochs.info, 
                          ch_type="eeg", 
                          units="Patterns (AU)", 
                          size=1.5).savefig(f"{result_dir}/{subject}_patterns.png")
    
    else:  # Statistical features
        # Compute statistical features
        X_features = np.concatenate([
            X.mean(axis=2),  # Mean
            X.var(axis=2),   # Variance
            # Skewness
            np.apply_along_axis(lambda a: np.mean((a - np.mean(a))**3) / np.std(a)**3, axis=2, arr=X),
            # Kurtosis
            np.apply_along_axis(lambda a: np.mean((a - np.mean(a))**4) / np.std(a)**4, axis=2, arr=X)
        ], axis=1)
        
        # Create the pipeline with StandardScaler
        if classifier == 'knn':
            clf_obj = KNeighborsClassifier(n_neighbors=5)
            clf = Pipeline([('scaler', StandardScaler()), ('knn', clf_obj)])
        elif classifier == 'svm':
            clf_obj = SVC(kernel='linear')
            clf = Pipeline([('scaler', StandardScaler()), ('svm', clf_obj)])
        elif classifier == 'lda':
            clf_obj = LinearDiscriminantAnalysis()
            clf = Pipeline([('scaler', StandardScaler()), ('lda', clf_obj)])
        elif classifier == 'mlp':
            clf_obj = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
            clf = Pipeline([('scaler', StandardScaler()), ('mlp', clf_obj)])
        
        # For statistical features, we use X_features instead of X
        X = X_features
        
        result_dir = f'./results/stats_{classifier}'
        os.makedirs(result_dir, exist_ok=True)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    # Evaluate the classifier
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)
    
    # Calculate performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    mean_scores = [np.mean(scores['test_' + metric]) for metric in metrics]
    std_scores = [np.std(scores['test_' + metric]) for metric in metrics]
    
    # Test inference performance on a separate test set
    # Use the last fold for performance testing
    cv_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(cv_test.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train the classifier for performance testing
    clf_perf = clone(clf)
    clf_perf.fit(X_train, y_train)
    
    # Measure inference performance
    performance_metrics = measure_inference_performance(clf_perf, X_test, y_test)
    
    # Measure model complexity
    complexity_metrics = test_model_complexity(clf_perf)
    
    # Print the results
    print(f"\nSubject {subject} - {feature_method.upper()} with {classifier.upper()}:")
    for metric, mean_score, std_score in zip(metrics, mean_scores, std_scores):
        print(f"{metric}: {mean_score:.3f} +/- {std_score:.3f}")
    
    # Print performance metrics
    print(f"Single prediction time: {performance_metrics['single_prediction_time_ms']['mean']:.2f} ± {performance_metrics['single_prediction_time_ms']['std']:.2f} ms")
    print(f"Real-time feasible: {performance_metrics['real_time_feasible']}")
    print(f"Max predictions/sec: {performance_metrics['max_predictions_per_second']:.1f}")
    print(f"Memory usage: {performance_metrics['memory_usage_mb']:.1f} MB")
    print(f"Model size: {complexity_metrics['model_size_kb']:.1f} KB")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance metrics plot
    ax1.bar(metrics, mean_scores, yerr=std_scores, capsize=5)
    ax1.set_ylabel('Score')
    ax1.set_title(f'Classification Performance')
    
    # Timing metrics plot
    timing_metrics = ['Single Pred.\n(ms)', 'Batch/Sample\n(ms)', 'Memory\n(MB)', 'Model Size\n(KB)']
    timing_values = [
        performance_metrics['single_prediction_time_ms']['mean'],
        performance_metrics['batch_time_per_sample_ms'],
        performance_metrics['memory_usage_mb'],
        complexity_metrics['model_size_kb']
    ]
    timing_errors = [
        performance_metrics['single_prediction_time_ms']['std'],
        0,  # No std for batch time per sample
        0,  # No std for memory usage
        0   # No std for model size
    ]
    
    ax2.bar(timing_metrics, timing_values, yerr=timing_errors, capsize=5, color='orange')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Real-time Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Subject {subject} - {feature_method.upper()} with {classifier.upper()}')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/{subject}_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Prepare results dictionary
    subject_results = {
        'classification_scores': {
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'metrics': metrics
        },
        'performance_metrics': performance_metrics,
        'complexity_metrics': complexity_metrics
    }
    
    return subject_results


def run_experiment(feature_methods, classifiers, subjects_range=(1, 110), task_type='execution'):
    """
    Run experiment for a combination of feature extraction methods and classifiers.
    
    Parameters:
    -----------
    feature_methods : list
        List of feature extraction methods to use
    classifiers : list
        List of classifiers to use
    subjects_range : tuple
        Range of subject IDs (start, end+1)
    task_type : str
        'execution' for motor execution or 'imagery' for motor imagery
    """
    for feature_method in feature_methods:
        for classifier in classifiers:
            print(f"\n\n{'='*80}")
            print(f"Running {feature_method.upper()} with {classifier.upper()}")
            print(f"{'='*80}")
            
            all_results = []
            subjects = []
            
            # Process each subject
            for subject in range(subjects_range[0], subjects_range[1]):
                try:
                    subject_results = process_subject(subject, feature_method, classifier, task_type)
                    all_results.append(subject_results)
                    subjects.append(subject)
                except Exception as e:
                    print(f"Error processing subject {subject}: {e}")
            
            # Create the result directory
            if feature_method == 'csp':
                result_dir = f'./results/csp_{classifier}'
            else:
                result_dir = f'./results/stats_{classifier}'
            
            os.makedirs(result_dir, exist_ok=True)
            
            # Format subjects for CSV
            formatted_subjects = [f"{s:03d}" for s in subjects]
            
            # Create classification performance DataFrame
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            row_index = pd.MultiIndex.from_product([metrics, ['mean', 'std']], names=['metric', 'statistic'])
            scores_df = pd.DataFrame(index=row_index, columns=formatted_subjects)
            
            # Create performance metrics DataFrame
            perf_metrics = [
                'single_prediction_time_ms_mean', 'single_prediction_time_ms_std',
                'batch_time_per_sample_ms', 'memory_usage_mb', 'cpu_usage_percent',
                'max_predictions_per_second', 'real_time_feasible', 'model_size_kb',
                'total_parameters'
            ]
            performance_df = pd.DataFrame(index=perf_metrics, columns=formatted_subjects)
            
            # Fill the DataFrames
            for i, (subject, result) in enumerate(zip(subjects, all_results)):
                subj_col = f"{subject:03d}"
                
                # Classification scores
                mean_scores = result['classification_scores']['mean_scores']
                std_scores = result['classification_scores']['std_scores']
                for j, metric in enumerate(metrics):
                    scores_df.loc[(metric, 'mean'), subj_col] = f"{mean_scores[j]:.3f}"
                    scores_df.loc[(metric, 'std'), subj_col] = f"{std_scores[j]:.3f}"
                
                # Performance metrics
                perf = result['performance_metrics']
                comp = result['complexity_metrics']
                performance_df.loc['single_prediction_time_ms_mean', subj_col] = f"{perf['single_prediction_time_ms']['mean']:.3f}"
                performance_df.loc['single_prediction_time_ms_std', subj_col] = f"{perf['single_prediction_time_ms']['std']:.3f}"
                performance_df.loc['batch_time_per_sample_ms', subj_col] = f"{perf['batch_time_per_sample_ms']:.3f}"
                performance_df.loc['memory_usage_mb', subj_col] = f"{perf['memory_usage_mb']:.1f}"
                performance_df.loc['cpu_usage_percent', subj_col] = f"{perf['cpu_usage_percent']:.1f}"
                performance_df.loc['max_predictions_per_second', subj_col] = f"{perf['max_predictions_per_second']:.1f}"
                performance_df.loc['real_time_feasible', subj_col] = str(perf['real_time_feasible'])
                performance_df.loc['model_size_kb', subj_col] = f"{comp['model_size_kb']:.1f}"
                performance_df.loc['total_parameters', subj_col] = str(comp['total_parameters'])
            
            # Save results to CSV
            scores_df.to_csv(f'{result_dir}/classification_scores.csv')
            performance_df.to_csv(f'{result_dir}/performance_metrics.csv')
            
            # Create summary statistics
            summary_stats = {}
            
            # Classification performance summary
            for metric in metrics:
                mean_values = [float(scores_df.loc[(metric, 'mean'), col]) for col in formatted_subjects]
                summary_stats[f'{metric}_mean'] = np.mean(mean_values)
                summary_stats[f'{metric}_std'] = np.std(mean_values)
            
            # Performance summary
            timing_values = [float(performance_df.loc['single_prediction_time_ms_mean', col]) for col in formatted_subjects]
            memory_values = [float(performance_df.loc['memory_usage_mb', col]) for col in formatted_subjects]
            size_values = [float(performance_df.loc['model_size_kb', col]) for col in formatted_subjects]
            feasible_count = sum([performance_df.loc['real_time_feasible', col] == 'True' for col in formatted_subjects])
            
            summary_stats['avg_prediction_time_ms'] = np.mean(timing_values)
            summary_stats['avg_memory_mb'] = np.mean(memory_values)
            summary_stats['avg_model_size_kb'] = np.mean(size_values)
            summary_stats['real_time_feasible_percentage'] = (feasible_count / len(formatted_subjects)) * 100
            
            # Save summary
            summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
            summary_df.to_csv(f'{result_dir}/summary_statistics.csv')
            
            print(f"Results saved to {result_dir}/")
            print(f"- Classification scores: classification_scores.csv")
            print(f"- Performance metrics: performance_metrics.csv")
            print(f"- Summary statistics: summary_statistics.csv")
            print(f"Real-time feasible: {feasible_count}/{len(formatted_subjects)} subjects ({summary_stats['real_time_feasible_percentage']:.1f}%)")


if __name__ == "__main__":
    # Define experimental parameters
    feature_methods = ['csp', 'stats']
    classifiers = ['knn', 'svm', 'lda', 'mlp']
    
    print("Running EEG decoding experiment with real-time performance assessment...")
    
    run_experiment(feature_methods, classifiers, subjects_range=(1, 110), task_type='execution')
    
    # Uncomment to run motor imagery experiment
    # run_experiment(feature_methods, classifiers, subjects_range=(1, 110), task_type='imagery')
    
    print("\nExperiment completed!")
    print("Check the results/ directory for:")
    print("- Classification performance scores")
    print("- Real-time performance metrics")
    print("- Summary statistics")
    print("- Individual subject plots showing both classification and timing performance") 