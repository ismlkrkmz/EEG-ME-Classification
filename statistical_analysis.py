import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Read CSV file
try:
    # Reading file - using semicolon separator and comma as decimal point
    df = pd.read_csv('accuracy-means.csv', sep=';', decimal=',')
    print("Data read successfully!")
    print(f"Data size: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
except Exception as e:
    print(f"File reading error: {e}")
    # Alternative path for manual data entry
    data = {
        'Classification Algorithm': ['KNN', 'LDA', 'MLP', 'SVM', 'KNN', 'LDA', 'MLP', 'SVM'],
        'Feature Extraction Method': ['CSP', 'CSP', 'CSP', 'CSP', 'Statistical Features', 'Statistical Features', 'Statistical Features', 'Statistical Features']
    }
    # Here we can manually enter the data
    df = pd.DataFrame(data)

# Create short names for models
df['Model'] = df['Classification Algorithm'] + '_' + df['Feature Extraction Method'].str.replace(' ', '_')

# Get accuracy values (excluding first two columns)
accuracy_columns = [col for col in df.columns if col not in ['Classification Algorithm', 'Feature Extraction Method', 'Model', 'Average']]

print(f"\nTotal number of subjects: {len(accuracy_columns)}")
print(f"Total number of models: {len(df)}")
print("\nModels:")
for i, model in enumerate(df['Model']):
    print(f"{i+1}. {model}")

# Get accuracy values for each model
models_data = {}
for idx, row in df.iterrows():
    model_name = row['Model']
    accuracies = []
    for col in accuracy_columns:
        try:
            # Convert comma-separated decimals to dot format
            val = str(row[col]).replace(',', '.')
            accuracies.append(float(val))
        except:
            print(f"Error: Invalid value in {col} column of {model_name} model: {row[col]}")
    models_data[model_name] = np.array(accuracies)

print(f"\nNumber of processed models: {len(models_data)}")

# Statistical test function
def statistical_test(acc1, acc2, model1_name, model2_name):
    """Applies paired t-test or Wilcoxon signed-rank test between two models"""
    
    # Calculate differences
    differences = acc1 - acc2
    
    # Normality test (Shapiro-Wilk)
    try:
        stat_norm, p_norm = shapiro(differences)
    except:
        p_norm = 0.01  # Reject normality assumption
    
    # Test selection
    if p_norm > 0.05:
        # If normal distribution, use paired t-test
        try:
            stat, p_value = ttest_rel(acc1, acc2)
            test_type = "Paired t-test"
        except:
            stat, p_value = wilcoxon(acc1, acc2, zero_method='zsplit')
            test_type = "Wilcoxon (t-test error)"
    else:
        # If no normal distribution, use Wilcoxon signed-rank test
        try:
            stat, p_value = wilcoxon(acc1, acc2, zero_method='zsplit')
            test_type = "Wilcoxon signed-rank"
        except:
            stat, p_value = ttest_rel(acc1, acc2)
            test_type = "Paired t-test (Wilcoxon error)"
    
    # Mean difference
    mean_diff = np.mean(differences)
    
    return {
        'test_type': test_type,
        'statistic': stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'normality_p': p_norm,
        'significant': p_value < 0.05
    }

# Comparison between all model pairs
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*80)

results = []
model_names = list(models_data.keys())

for i, model1 in enumerate(model_names):
    for j, model2 in enumerate(model_names):
        if i < j:  # Test each pair only once
            acc1 = models_data[model1]
            acc2 = models_data[model2]
            
            result = statistical_test(acc1, acc2, model1, model2)
            result['model1'] = model1
            result['model2'] = model2
            result['mean1'] = np.mean(acc1)
            result['mean2'] = np.mean(acc2)
            
            results.append(result)

# Show results
print(f"\nTotal {len(results)} model pairs compared.\n")

# Significant differences
significant_results = [r for r in results if r['significant']]
print(f"Number of pairs showing statistically significant differences: {len(significant_results)}")

if significant_results:
    print("\nSTATISTICALLY SIGNIFICANT DIFFERENCES:")
    print("-" * 60)
    
    for result in significant_results:
        print(f"\n{result['model1']} vs {result['model2']}")
        print(f"  Average accuracy: {result['mean1']:.4f} vs {result['mean2']:.4f}")
        print(f"  Mean difference: {result['mean_difference']:.4f}")
        print(f"  Test type: {result['test_type']}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  {'✓' if result['significant'] else '✗'} p < 0.05")

# Detailed results table
print("\n" + "="*120)
print("DETAILED RESULTS TABLE")
print("="*120)
print(f"{'Model 1':<25} {'Model 2':<25} {'Avg1':<8} {'Avg2':<8} {'Diff':<8} {'Test':<15} {'p-value':<12} {'Significant':<12}")
print("-" * 120)

for result in results:
    significance = "✓" if result['significant'] else "✗"
    print(f"{result['model1']:<25} {result['model2']:<25} "
          f"{result['mean1']:<8.4f} {result['mean2']:<8.4f} "
          f"{result['mean_difference']:<8.4f} {result['test_type']:<15} "
          f"{result['p_value']:<12.6f} {significance:<12}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

model_means = {name: np.mean(data) for name, data in models_data.items()}
sorted_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)

print("\nModel performance ranking (average accuracy):")
for i, (model, mean_acc) in enumerate(sorted_models, 1):
    print(f"{i}. {model}: {mean_acc:.4f}")

print(f"\nTotal number of comparisons: {len(results)}")
print(f"Number of significant differences: {len(significant_results)}")
print(f"Percentage of significant differences: {len(significant_results)/len(results)*100:.1f}%")

# Bonferroni correction
alpha_bonferroni = 0.05 / len(results)
bonferroni_significant = [r for r in results if r['p_value'] < alpha_bonferroni]

print(f"\nAfter Bonferroni correction (α = {alpha_bonferroni:.6f}):")
print(f"Number of significant differences: {len(bonferroni_significant)}")

if bonferroni_significant:
    print("\nSignificant differences after Bonferroni correction:")
    for result in bonferroni_significant:
        print(f"  {result['model1']} vs {result['model2']}: p = {result['p_value']:.6f}")