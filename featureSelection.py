import math
from collections import Counter

def compute_variance(values):
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n

def variance_threshold_select(X, feature_names, threshold=0.1):
    """
    Selects features whose variance is >= threshold.
    """
    selected_features = []
    scores = {}
    
    n_features = len(X[0])
    for j in range(n_features):
        column = [row[j] for row in X]
        var = compute_variance(column)
        scores[feature_names[j]] = var
        if var >= threshold:
            selected_features.append(feature_names[j])
            
    return selected_features, scores

def compute_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / math.sqrt(denom_x * denom_y)

def correlation_select(X, y, feature_names, top_k=2):
    """
    Ranks features by absolute Pearson correlation with target y and selects top_k.
    """
    scores = {}
    for j in range(len(X[0])):
        column = [row[j] for row in X]
        r = compute_correlation(column, y)
        scores[feature_names[j]] = abs(r)
        
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [feat for feat, score in ranked[:top_k]]
    return selected, scores


# =====================================================================
# 3. CHI-SQUARE (chi2) TEST
# Measures independence between non-negative feature frequencies and target classes.
# Formula: Chi2 = sum((Observed - Expected)^2 / Expected)
# =====================================================================

def chi_square_select(X, y, feature_names, top_k=2):
    """
    Computes Chi-Square statistics for non-negative numerical/frequency features.
    """
    classes = set(y)
    total_samples = len(y)
    class_counts = Counter(y)
    
    scores = {}
    for j in range(len(X[0])):
        column = [row[j] for row in X]
        total_feature_sum = sum(column)
        
        if total_feature_sum == 0:
            scores[feature_names[j]] = 0.0
            continue
            
        chi2_stat = 0.0
        for c in classes:
            # Observed sum of feature j for samples in class c
            observed = sum(column[i] for i in range(total_samples) if y[i] == c)
            # Expected sum assuming independence
            expected = total_feature_sum * (class_counts[c] / total_samples)
            
            if expected > 0:
                chi2_stat += ((observed - expected) ** 2) / expected
                
        scores[feature_names[j]] = chi2_stat
        
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [feat for feat, score in ranked[:top_k]]
    return selected, scores


# =====================================================================
# 4. INFORMATION GAIN (MUTUAL INFORMATION)
# Measures reduction in entropy of target Y given feature X.
# Formula: IG(Y, X) = H(Y) - H(Y|X)
# =====================================================================

def compute_entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def information_gain_select(X, y, feature_names, top_k=2):
    """
    Computes Information Gain for discrete/categorical features against target y.
    """
    base_entropy = compute_entropy(y)
    n_samples = len(y)
    scores = {}
    
    for j in range(len(X[0])):
        column = [row[j] for row in X]
        unique_values = set(column)
        
        conditional_entropy = 0.0
        for val in unique_values:
            sub_y = [y[i] for i in range(n_samples) if column[i] == val]
            prob_val = len(sub_y) / n_samples
            conditional_entropy += prob_val * compute_entropy(sub_y)
            
        ig = base_entropy - conditional_entropy
        scores[feature_names[j]] = ig
        
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [feat for feat, score in ranked[:top_k]]
    return selected, scores


# =====================================================================
# DEMONSTRATION & TESTING
# =====================================================================

if __name__ == "__main__":
    # Sample Dataset: 6 samples, 4 numerical features, 1 categorical feature
    # Feature 3 is constant (zero variance)
    X_num = [
        [1.2, 0.5, 0.0],
        [1.0, 0.4, 0.0],
        [0.8, 0.6, 0.0],
        [8.1, 7.8, 0.0],
        [7.9, 8.2, 0.0],
        [8.0, 8.0, 0.0]
    ]
    y_class = [0, 0, 0, 1, 1, 1]
    num_names = ["Age_Scaled", "Income_Scaled", "Constant_Feature"]

    X_cat = [
        ["Small", "Red"],
        ["Small", "Red"],
        ["Small", "Blue"],
        ["Large", "Blue"],
        ["Large", "Blue"],
        ["Large", "Blue"]
    ]
    cat_names = ["Size", "Color"]

    print("--- 1. Variance Threshold (Threshold = 0.01) ---")
    selected_var, scores_var = variance_threshold_select(X_num, num_names, threshold=0.01)
    print(f"Scores: {scores_var}")
    print(f"Selected: {selected_var}\n")

    print("--- 2. Pearson Correlation (Top 2 Features) ---")
    selected_corr, scores_corr = correlation_select(X_num, y_class, num_names, top_k=2)
    print(f"Absolute r Scores: {scores_corr}")
    print(f"Selected: {selected_corr}\n")

    print("--- 3. Chi-Square Test (Top 2 Non-Negative Features) ---")
    selected_chi2, scores_chi2 = chi_square_select(X_num, y_class, num_names, top_k=2)
    print(f"Chi2 Scores: {scores_chi2}")
    print(f"Selected: {selected_chi2}\n")

    print("--- 4. Information Gain (Top 2 Categorical Features) ---")
    selected_ig, scores_ig = information_gain_select(X_cat, y_class, cat_names, top_k=2)
    print(f"IG Scores: {scores_ig}")
    print(f"Selected: {selected_ig}")