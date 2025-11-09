"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon



concrete_path = "../datasets/concrete_strength.csv"
material_path = "../datasets/material_properties.csv"
loads_path = "../datasets/structural_loads.csv"

#Data loading and exploration
def load_data(concrete_path,material_path,loads_path):
    try:
        df_concrete = pd.read_csv(concrete_path)
        df_material = pd.read_csv(material_path)
        df_loads = pd.read_csv(loads_path)

        for (name,df) in ("Concrete Strength", df_concrete),("Structural Loads", df_loads),  ("Material Properties",df_material):
            print("="*50)
            print(f"\n--- Basic Informations: {name.upper()} Dataset ---")
            print(f"\nShape: {df.shape}")
            column_list = df.columns.tolist()
            print(f"\nColumn Names and Data Types:")
            print(column_list)
            print()
            print(df.dtypes)
            print()

            initial_rows = len(df)
            df.dropna(inplace=True)
            dropped_rows = initial_rows - len(df)

            print("--- Missing Value Handling ---")
            print(f"{name.upper()} Dataset: Dropped {dropped_rows} rows (Remaining: {len(df)} rows)\n")

            print("--- Summary Statistics ---\n")
            print(df.head(5))
            print()
            print(df.describe())

        return  df_concrete,df_material,df_loads


    except FileNotFoundError:
        print("Error: One or more dataset files not found.")
        return None, None, None
    #Captures any error, and returns
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None, None, None
    pass


#Measures of central tendency
#Task-1
def calculate_descriptive_stats(df_concrete):
    """Calculate all descriptive statistics."""
    #Concrete concrete strength statistics


    mean_val = df_concrete['strength_mpa'].mean()
    median_val = df_concrete['strength_mpa'].median()
    #mode()[0]--> gives the most repetitive strength value.
    mode_val = df_concrete['strength_mpa'].mode()[0]
    std_val = df_concrete['strength_mpa'].std()
    var_val = df_concrete['strength_mpa'].var()
    skew_val = df_concrete['strength_mpa'].skew()
    kurtosis_val = df_concrete['strength_mpa'].kurtosis()

    print(f"\n=== Concrete Strength Statistics ===\n")
    print(f"Mean Concrete Strength (MPa)   : {mean_val:.2f}")
    print(f"Median Concrete Strength (MPa) : {median_val:.2f}")
    print(f"Mode Concrete Strength (MPa)   : {mode_val:.2f}")
    print(f"Standard Deviation : {std_val:.2f}")
    print(f"Variance           : {var_val:.2f}")
    print(f"Skewness           : {skew_val:.2f}")
    print(f"Kurtosis           : {kurtosis_val:.2f}")

    print("\n=== Assessing the Shape of the Concrete Strength Distribution ===")

    if mean_val > median_val > mode_val:
        print("\nMean > Median > Mode: Concrete Strength Distribution is **Right-Skewed**.")
        print("When Each Measure is Appropriate:")
        print("- Mean: Sensitive to high outliers, not ideal here since it overstates typical strength.")
        print("- Median: Best representation of typical concrete strength; resistant to outliers.")
        print("- Mode: Shows the most frequently produced concrete strength class (e.g., standard mix).")
        print("For this dataset, **Median** is the most reliable measure.\n")
    elif mean_val < median_val < mode_val:
        print("\nMean < Median < Mode: Concrete Strength Distribution is **Left-Skewed**.")
        print("When Each Measure is Appropriate:")
        print("- Mean: Distorted by low outliers; underestimates the typical concrete strength.")
        print("- Median: More robust and better reflects central tendency.")
        print("- Mode: Represents the most common strength value used in design mixes.")
        print("For this dataset, **Median** remains the most appropriate.\n")
    elif mean_val == median_val == mode_val:
        print("\nMean = Median = Mode: Concrete Strength Distribution is **Perfectly Symmetric**.")
        print("Mean = Median = Mode: values are evenly distributed around the center.")
    else:
        pass

    print(f"Skewness value: {skew_val:.2f}")
    if skew_val > 0.5:
        print("   → The data is **moderately to strongly right-skewed** (long right tail) Skewness value > 0.5.")
    elif skew_val < -0.5:
        print("   → The data is **moderately to strongly left-skewed** (long left tail) Skewness value < - 0.5.")
    else:
        print("   → The data is **fairly symmetric** (little skew).")

    print(f"\nKurtosis value: {kurtosis_val:.2f}")
    if kurtosis_val > 3:
        print("   → The distribution is **Leptokurtic** — sharply peaked with heavy tails.")
        print("     This means there are more extreme strength values than a normal distribution.")
    elif kurtosis_val < 3:
        print("   → The distribution is **Platykurtic** — flatter than a normal curve.")
        print("     This indicates less extreme variation and a more uniform spread of strength values.")
    else:
        print("   → The distribution is **Mesokurtic**, similar to a normal distribution.")


    print(f"\nStandard Deviation : {std_val:.2f}")
    print(f"Variance           : {var_val:.2f}")
    print("→ A higher standard deviation or variance indicates that the concrete strengths vary widely, while lower values suggest a more consistent quality of concrete.")

    print("\n=== Design & Quality Interpretation ===\n")

    if skew_val > 0.5:
        print("The dataset is Right-Skewed → few high-strength outliers.")
        print("- For **design specifications**, use the MEDIAN or CHARACTERISTIC strength rather than the mean.")
        print("- The mean (", round(mean_val, 2), "MPa) may **overestimate** actual field performance.")
        print("- Median (", round(median_val, 2), "MPa) better reflects typical concrete behavior.")
        print(
            "- Quality control should monitor high-strength deviations — may indicate inconsistent curing or mix proportion.")
    elif abs(skew_val) <= 0.5:
        print("The distribution is approximately Symmetric.")
        print("- Mean and median are similar → process is stable.")
        print("- The mean (", round(mean_val, 2), "MPa) can be used for **design strength** and quality assessment.")
        print("- Low skewness indicates consistent batching and curing.")
    else:
        print("The dataset is Left-Skewed → few low-strength batches.")
        print("- For **design safety**, use the LOWER TAIL (5th percentile) or median.")
        print("- Mean may **underestimate** potential capacity; investigate weak mixes or curing problems.")

    # Check standard deviation (process variability)
    if std_val > 5:
        print("\nHigh standard deviation (σ =", round(std_val, 2), "): strength varies widely.")
        print("- Indicates variable quality; mix control or curing inconsistency.")
        print("- Implement stricter batching or material quality monitoring.")
    else:
        print("\nLow standard deviation (σ =", round(std_val, 2), "): production is uniform.")
        print("- Concrete quality meets design repeatability standards.")

    # Kurtosis (consistency and reliability)
    if kurtosis_val > 3:
        print("\nHigh kurtosis (", round(kurtosis_val, 2), "): results are tightly clustered — good consistency.")
        print("- Indicates reliable quality control and predictable performance.")
    elif kurtosis_val < 3:
        print("\nLow kurtosis (", round(kurtosis_val, 2), "): wider spread — monitor variability.")
        print("- Suggests occasional outliers; quality process can be optimized.")


    plt.figure(figsize=(10, 6))
    sns.histplot(df_concrete['strength_mpa'], bins=25, color='skyblue', edgecolor='white', alpha=0.7, stat="density")

    mean_val = df_concrete['strength_mpa'].mean()
    std_val = df_concrete['strength_mpa'].std()

    x = np.linspace(df_concrete['strength_mpa'].min(), df_concrete['strength_mpa'].max(), 100)
    y = norm.pdf(x, mean_val, std_val)
    plt.plot(x, y, color='darkred', linewidth=2, label='Normal Curve')

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean ({mean_val:.2f})")
    plt.axvline(df_concrete['strength_mpa'].median(), color='green', linestyle='-', linewidth=2, label="Median")
    plt.axvline(df_concrete['strength_mpa'].mode()[0], color='purple', linestyle=':', linewidth=2, label="Mode")

    plt.title("Histogram with Normal Curve Overlay", fontsize=13, fontweight='bold')
    plt.xlabel("Concrete Strength (MPa)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.show()
    plt.close()

    #Calculating quartiles
    Q1 = df_concrete['strength_mpa'].quantile(0.25)
    Q2 = df_concrete['strength_mpa'].quantile(0.50)
    Q3 = df_concrete['strength_mpa'].quantile(0.75)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_concrete['strength_mpa'], color='skyblue')

    plt.title("Boxplot of Concrete Strength (MPa)", fontsize=13, fontweight='bold')
    plt.xlabel("Concrete Strength (MPa)")
    plt.grid(alpha=0.3, linestyle='--')

    plt.text(Q1, 0.05, f"Q1 = {Q1:.2f}", rotation=90, verticalalignment='bottom', color='blue', fontsize=10, fontweight='bold')
    plt.text(Q2, 0.05, f"Median (Q2) = {Q2:.2f}", rotation=90, verticalalignment='bottom', color='green', fontsize=10, fontweight='bold')
    plt.text(Q3, 0.05, f"Q3 = {Q3:.2f}", rotation=90, verticalalignment='bottom', color='red', fontsize=10, fontweight='bold')

    plt.savefig("Boxplot of Concrete Strength (MPa).png")
    plt.show()
    plt.close()


    return mean_val, median_val, mode_val, std_val, var_val , skew_val , kurtosis_val

#Probability Modeling

def calculate_probability_binomial(n, p):
    """Calculate binomial probabilities."""
    n = 100  #Total Tests
    p = 0.05 #Defect Rate

    #The probability of having exactly 3 defects
    prob_exactly_3 = binom.pmf(k=3, n=n, p=p)
    prob_at_most_5 = binom.cdf(k=5, n=n, p=p)

    print("\n--- Quality control: 100 components tested, 5% defect rate. ---")
    print("\n--- What is probability of exactly 3 defective components? ---")
    print(f"P(X=3): {prob_exactly_3:.3f}")
    print("\n--- What is probability of less than 5 defective components? ---")
    print(f"P(X<=5): {prob_at_most_5:.3f}")
    # 4. PLOT PMF AND CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k_values = np.arange(binom.ppf(0.001, n, p), binom.ppf(0.999, n, p) + 1)

    # Plot 1: Probability Mass Function (PMF)
    axes[0].bar(k_values, binom.pmf(k_values, n, p), color='skyblue')
    axes[0].set_title('Binomial Probability Mass Function (PMF)')
    axes[0].set_xlabel('Number of Defects (k)')
    axes[0].set_ylabel('Probability')
    axes[0].grid(axis='y', alpha=0.5)

    # Plot 2: Cumulative Distribution Function (CDF)
    axes[1].step(k_values, binom.cdf(k_values, n, p), color='darkred', where='post')
    axes[1].set_title('Binomial Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Number of Defects (k)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # 5. GÖRSELİ KAYDETME
    plt.savefig("binomial_distribution_analysis.png")
    plt.show()
    plt.close()
    print("\nVisual analysis saved to: binomial_distribution_analysis.png")

    return prob_exactly_3,prob_at_most_5

def calculate_probability_normal(mean, std):
    """Calculate normal probabilities."""
    mean = 250
    std = 15
    variance = std ** 2

    #Exceeding 280 mpa
    prob_exceeds_280 = 1 - norm.cdf(280, loc=mean, scale=std)
    percentage_exceeds_280 = prob_exceeds_280 * 100

    #Finding 95th percentile
    percentile_95 = norm.ppf(0.95, loc=mean, scale=std)

    print("\n--- Steel yield strength: Mean = 250 MPa, Std = 15 MPa. ---")
    print("\n--- What percentage exceeds 280 MPa? ---")
    print(f" ={percentage_exceeds_280:.3f}")
    print("\n--- What is the 95th percentile? ---")
    print(f" ={percentile_95:.3f}")
    # 1. APPLY TO A REAL ENGINEERING SCENARIO (Probabilities)

    # Exceeding 280 MPa: P(X > 280) = 1 - P(X <= 280)
    prob_exceeds_280 = 1 - norm.cdf(280, loc=mean, scale=std)
    percentage_exceeds_280 = prob_exceeds_280 * 100

    # Finding 95th percentile
    percentile_95 = norm.ppf(0.95, loc=mean, scale=std)

    # 2. CALCULATE MEAN AND VARIANCE (Expected Value)
    mean_expected = norm.mean(loc=mean, scale=std)
    variance_expected = norm.var(loc=mean, scale=std)

    # 3. GENERATE RANDOM SAMPLES (Simulation)
    N_SAMPLES = 10000  # Generate a large number of simulations
    random_samples = norm.rvs(loc=mean, scale=std, size=N_SAMPLES)

    # Calculate sample moments for comparison
    sample_mean = np.mean(random_samples)
    sample_var = np.var(random_samples)

    # --- CONSOLE OUTPUTS ---
    print("\n--- Normal Analysis: Steel Yield Strength (mu=250, sigma=15) ---")

    print("\n[Probability Calculations]")
    print(f"Percentage Exceeding 280 MPa: {percentage_exceeds_280:.3f}%")
    print(f"95th Percentile (MPa): {percentile_95:.3f}")

    print("\n[Distribution Moments]")
    print(f"Theoretical Mean (E[X]): {mean_expected:.2f}")
    print(f"Theoretical Variance (Var[X]): {variance_expected:.2f}")
    print(f"Simulated Mean: {sample_mean:.2f}")
    print(f"Simulated Variance: {sample_var:.2f}")

    # 4. PLOT PDF AND CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define the range for plotting (e.g., Mean +/- 3*Std Dev)
    x_min = mean - 3.5 * std
    x_max = mean + 3.5 * std
    x = np.linspace(x_min, x_max, 200)

    # Plot 1: Probability Density Function (PDF)
    axes[0].plot(x, norm.pdf(x, loc=mean, scale=std), color='darkred', linewidth=2)
    axes[0].set_title('Normal Probability Density Function (PDF)')
    axes[0].set_xlabel('Yield Strength (MPa)')
    axes[0].set_ylabel('Density')
    axes[0].grid(axis='y', alpha=0.5)

    # Plot 2: Cumulative Distribution Function (CDF)
    axes[1].plot(x, norm.cdf(x, loc=mean, scale=std), color='darkblue', linewidth=2)
    axes[1].set_title('Normal Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Yield Strength (MPa)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # 5. SAVE THE VISUALIZATION
    plt.savefig("normal_distribution_analysis.png")
    plt.show()
    plt.close()
    print("\nVisual analysis saved to: normal_distribution_analysis.png")

    return percentage_exceeds_280, percentile_95

def calculate_probability_poisson(lambda_param):
    """Calculate Poisson probabilities."""
    #Average passing of heavy cars
    mu = 10
    prob_exactly_8 = poisson.pmf(k=8, mu=mu)

    # P(X > 15) = 1 - P(X <= 15)
    prob_more_than_15 = 1 - poisson.cdf(k=15, mu=mu)

    print("\n--- Bridge load events: Average 10 heavy trucks per hour. ---")
    print("\n--- What is probability of exactly 8 trucks in an hour? ---")
    print(f"P(X=8): {prob_exactly_8:.3f}")
    print("\n--- What is probability of > 15 trucks in an hour? ---")
    print(f"P(X>15): {prob_more_than_15:.3f}")

    # 2. CALCULATE MEAN AND VARIANCE (Expected Value)
    mean_expected = poisson.mean(mu)
    variance_expected = poisson.var(mu)

    # 3. GENERATE RANDOM SAMPLES (Simulation)
    N_SAMPLES = 10000  # Generate a large number of simulations
    random_samples = poisson.rvs(mu=mu, size=N_SAMPLES)

    # Calculate sample moments for comparison
    sample_mean = np.mean(random_samples)
    sample_var = np.var(random_samples)

    # --- CONSOLE OUTPUTS ---
    print("\n--- Poisson Analysis: Bridge Load Events (lambda=10) ---")

    print("\n[Probability Calculations]")
    print(f"P(X=8) (Exactly 8 trucks): {prob_exactly_8:.4f}")
    print(f"P(X>15) (More than 15 trucks): {prob_more_than_15:.4f}")

    print("\n[Distribution Moments]")
    print(f"Theoretical Mean (E[X]): {mean_expected:.2f}")
    print(f"Theoretical Variance (Var[X]): {variance_expected:.2f}")
    print(f"Simulated Mean: {sample_mean:.2f}")
    print(f"Simulated Variance: {sample_var:.2f}")

    # 4. PLOT PMF AND CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define the range for plotting (k values)
    k_values = np.arange(poisson.ppf(0.001, mu), poisson.ppf(0.999, mu) + 1)

    # Plot 1: Probability Mass Function (PMF)
    axes[0].bar(k_values, poisson.pmf(k_values, mu), color='skyblue')
    axes[0].set_title('Poisson Probability Mass Function (PMF)')
    axes[0].set_xlabel('Number of Events (k)')
    axes[0].set_ylabel('Probability')
    axes[0].grid(axis='y', alpha=0.5)

    # Plot 2: Cumulative Distribution Function (CDF)
    axes[1].step(k_values, poisson.cdf(k_values, mu), color='darkred', where='post')
    axes[1].set_title('Poisson Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Number of Events (k)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()

    # 5. SAVE THE VISUALIZATION
    plt.savefig("poisson_distribution_analysis.png")
    plt.show()
    plt.close()
    return prob_exactly_8, prob_more_than_15


def calculate_probability_exponential(mean):
    """Calculate exponential probabilities."""
    mean_lifetime = 1000
    scale_param = mean_lifetime

    # Probability of failure before 500 hours
    prob_failure_before_500 = expon.cdf(500, loc=0, scale=scale_param)

    # Probability of surviving beyond 1500 hours (1 - CDF)
    prob_survival_beyond_1500 = 1 - expon.cdf(1500, loc=0, scale=scale_param)

    print("\n--- Component lifetime: Mean = 1000 hours. ---")
    print("\n--- What is probability of failure before 500 hours? ---")
    print(f"P(X<500): {prob_failure_before_500:.3f}")
    print("\n--- What is probability of surviving beyond 1500 hours? ---")
    print(f"P(X>1500): {prob_survival_beyond_1500:.3f}")

    # 2. CALCULATE MEAN AND VARIANCE (Expected Value)
    mean_expected = expon.mean(loc=0, scale=scale_param)
    variance_expected = expon.var(loc=0, scale=scale_param)

    # 3. GENERATE RANDOM SAMPLES (Simulation)
    N_SAMPLES = 10000
    random_samples = expon.rvs(loc=0, scale=scale_param, size=N_SAMPLES)

    # Calculate sample moments for comparison
    sample_mean = np.mean(random_samples)
    sample_var = np.var(random_samples)

    # --- CONSOLE OUTPUTS ---
    print("\n--- Exponential Analysis: Component Lifetime (Mean=1000 hours) ---")

    print("\n[Probability Calculations]")
    print(f"P(X<500) (Failure before 500h): {prob_failure_before_500:.4f}")
    print(f"P(X>1500) (Surviving beyond 1500h): {prob_survival_beyond_1500:.4f}")

    print("\n[Distribution Moments]")
    print(f"Theoretical Mean (E[X]): {mean_expected:.2f}")
    print(f"Theoretical Variance (Var[X]): {variance_expected:.2f}")
    print(f"Simulated Mean: {sample_mean:.2f}")
    print(f"Simulated Variance: {sample_var:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Define the range for plotting (x-axis)
    x_max = 4 * mean_lifetime  # 4000 saate kadar çizim yapar
    x = np.linspace(0, x_max, 200)

    # Plot 1: Probability Density Function (PDF)
    axes[0].plot(x, expon.pdf(x, loc=0, scale=scale_param), color='darkred', linewidth=2)
    axes[0].set_title('Exponential Probability Density Function (PDF)')
    axes[0].set_xlabel('Lifetime (Hours)')
    axes[0].set_ylabel('Density')
    axes[0].grid(axis='y', alpha=0.5)

    # Plot 2: Cumulative Distribution Function (CDF)
    axes[1].plot(x, expon.cdf(x, loc=0, scale=scale_param), color='darkblue', linewidth=2)
    axes[1].set_title('Exponential Cumulative Distribution Function (CDF)')
    axes[1].set_xlabel('Lifetime (Hours)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(axis='y', alpha=0.5)

    plt.tight_layout()
    # 5. SAVE THE VISUALIZATION
    plt.savefig("exponential_distribution_analysis.png")
    plt.show()
    plt.close()
    return prob_failure_before_500, prob_survival_beyond_1500


def apply_bayes_theorem(prior, sensitivity, specificity):
    """Apply Bayes' theorem for diagnostic test scenario."""
    # Base Rate: %5 structures are damaged.
    P_D = 0.05
    # %95 structures are not damaged
    P_Dc = 1 - P_D

    # If damaged, it is detected %95. (Sensitivity)
    P_T_given_D = 0.95
    # If not damaged, test is "false damaged" %10. (False Positive Rate)
    P_T_given_Dc = 0.10

    # False negative rate: P(T-|D)
    P_Tc_given_D = 1 - P_T_given_D

    # Specivity: test gives negative when there is no damage: P(T-|Dc)
    P_Tc_given_Dc = 1 - P_T_given_Dc

    # Joint Probabilities
    P_D_and_T = P_T_given_D * P_D  # True Positive
    P_Dc_and_T = P_T_given_Dc * P_Dc  # False Positive

    P_D_and_Tc = P_Tc_given_D * P_D  # False Negative
    P_Dc_and_Tc = P_Tc_given_Dc * P_Dc  # True Negative

    # Bayes Theorem Results
    P_T_positive = P_D_and_T + P_Dc_and_T
    P_D_given_T = P_D_and_T / P_T_positive


    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    def draw_node(x, y, label, color='lightblue', box=True):
        if box:
            plt.plot(x, y, 'o', markersize=15, color=color, zorder=3)
        plt.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)

    def draw_edge(x1, y1, x2, y2, label, color='gray'):
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, zorder=1)
        # Etiket, çizginin ortasına hafifçe kaydırılarak yerleştirilir
        plt.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, ha='center', va='center',
                 fontsize=8, color='darkgreen',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=2)

    # Starting Level
    draw_node(0, 0, "START", 'lightgray', box=False)

    # Level 1
    draw_node(1.5, 1.5, f"Damage", 'salmon')
    draw_node(1.5, -1.5, f"No Damage", 'lightgreen')

    # Seviye 2: Test Sonucu (Pozitif / Negatif)
    # Hasar dalı
    draw_node(3.5, 2.0, f"T+", 'skyblue')  # True Positive Path
    draw_node(3.5, 1.0, f"T-", 'darkgray')  # False Negative Path
    # Hasar Yok dalı
    draw_node(3.5, -1.0, f"T+", 'skyblue')  # False Positive Path
    draw_node(3.5, -2.0, f"T-", 'darkgray')  # True Negative Path

    # 0 -> 1 (Prior Probabilities)
    draw_edge(0, 0, 1.5, 1.5, f"{P_D:.2f}", 'salmon')
    draw_edge(0, 0, 1.5, -1.5, f"{P_Dc:.2f}", 'lightgreen')

    # 1 -> 2
    draw_edge(1.5, 1.5, 3.5, 2.0, f"{P_T_given_D:.2f}", 'blue')
    draw_edge(1.5, 1.5, 3.5, 1.0, f"{P_Tc_given_D:.2f}", 'red')

    draw_edge(1.5, -1.5, 3.5, -1.0, f"{P_T_given_Dc:.2f}", 'red')
    draw_edge(1.5, -1.5, 3.5, -2.0, f"{P_Tc_given_Dc:.2f}", 'blue')

    plt.text(4.5, 2.0, f"P(D ∩ T+) TP: {P_D_and_T:.4f}", fontsize=10, ha='left', color='blue', fontweight='bold')
    plt.text(4.5, 1.0, f"P(D ∩ T-) FN: {P_D_and_Tc:.4f}", fontsize=10, ha='left', color='red')
    plt.text(4.5, -1.0, f"P(Dc ∩ T+) FP: {P_Dc_and_T:.4f}", fontsize=10, ha='left', color='red', fontweight='bold')
    plt.text(4.5, -2.0, f"P(Dc ∩ T-) TN: {P_Dc_and_Tc:.4f}", fontsize=10, ha='left', color='blue')

    plt.text(1.7, 4.0,
             f"BAYES THEOREM RESULTS\n"
             f"P(D | T+) = P(D ∩ T+) / P(T+)\n"
             f"= {P_D_and_T:.4f} / ({P_D_and_T:.4f} + {P_Dc_and_T:.4f})\n"
             f"= {P_D_and_T:.4f} / {P_T_positive:.4f}\n"
             f"= {P_D_given_T:.4f} ({P_D_given_T * 100:.2f}%)",
             fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5, ec="black"))

    plt.xlim(-0.5, 7.5)
    plt.ylim(-3.0, 5.0)
    plt.axis('off')

    plt.savefig("Bayes Theorem Results.png")
    plt.show()
    plt.close()

    print("\n--- If test is positive, what is probability of actual damage? ---")
    print(f"If test is positive, the probability of actual damage is: ({P_D_given_T * 100:.2f}%)")
    print(
        "Despite the test's high accuracy, the low prior damage rate (5%) leads to a significant risk of False Positives.")
    print("Implications:")
    print(
        f"1. **High False Positive Risk:** Approximately two-thirds ({1 - P_D_given_T:.2f}) of positive test results are likely false alarms.")
    print("2. **Decision Rule:** Engineers should NOT immediately schedule repairs based on a single positive test.")
    print(
        "3. **Required Action:** A positive result should only serve to prioritize the structure for a mandatory SECONDARY, more reliable, visual or invasive inspection, to prevent unnecessary cost and resource waste.")

    pass


#Task-2: material comparison

def plot_material_comparison(df_material):
        """
        Calculates statistics, creates comparative boxplots, and interprets material variability
        for 'yield_strength_mpa' across different 'material_type' categories.
        """

        # 1. Calculate Statistics for each material type
        statical_summary = df_material.groupby("material_type")["yield_strength_mpa"].describe()

        # Set Pandas options to display all columns/rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print("\n--- Statistical Summary (Yield Strength per Material) ---")
        print(statical_summary)

        # 2. Create Comparative Boxplots
        plt.figure(figsize=(12, 7))

        sns.boxplot(
            x='material_type',
            y='yield_strength_mpa',
            data=df_material,
            hue='material_type',
            palette='Set2',
            legend=False
        )
        plt.title('Yield Strength Comparison Across Material Types', fontsize=16)
        plt.xlabel('Material Type')
        plt.ylabel('Yield Strength (MPa)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--')


        # Q1, Q2 (Medyan) ve Q3 değerlerini hesapla
        quartiles = df_material.groupby('material_type')['yield_strength_mpa'].quantile(
            [0.25, 0.5, 0.75]).unstack().reset_index()
        quartiles.columns = ['material_type', 'Q1', 'Q2_Median', 'Q3']

        # IQR hesapla
        quartiles['IQR'] = quartiles['Q3'] - quartiles['Q1']

        # Etiketler için X pozisyonlarını al
        material_types = df_material['material_type'].unique()
        x_positions = {material: i for i, material in enumerate(material_types)}

        # Dikey aralık referansı
        y_range = df_material['yield_strength_mpa'].max() - df_material['yield_strength_mpa'].min()

        for index, row in quartiles.iterrows():
            material = row['material_type']
            x_pos = x_positions[material]

            # Q1 Label (Kutunun altı)
            plt.text(x_pos, row['Q1'], f'Q1: {row["Q1"]:.1f}',
                     ha='center', va='top', fontsize=9, color='darkgreen',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

            # Q2 (Medyan) Label (Kutunun ortası)
            plt.text(x_pos, row['Q2_Median'], f'Q2: {row["Q2_Median"]:.1f}',
                     ha='center', va='center', fontsize=10, color='darkblue', fontweight='bold',
                     bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

            # Q3 Label (Kutunun üstü)
            plt.text(x_pos, row['Q3'], f'Q3: {row["Q3"]:.1f}',
                     ha='center', va='bottom', fontsize=9, color='darkgreen',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1'))

            # IQR Label (Q3'ün biraz üstüne yerleştirilir)
            iqr_label_pos_y = row['Q3'] + 0.04 * y_range
            plt.text(x_pos, iqr_label_pos_y, f'IQR: {row["IQR"]:.1f}',
                     ha='center', va='bottom', fontsize=9, color='black',
                     bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
        # 3. Compare Means and Standard Deviations

        print("\n--- Comparison and Interpretation ---")

        # Extract Means and Standard Deviations
        means = statical_summary['mean'].sort_values(ascending=False)
        stds = statical_summary['std'].sort_values(ascending=False)

        highest_mean_material = means.index[0]
        highest_std_material = stds.index[0]
        plt.savefig("material_comparison_boxplot.png")
        plt.show()
        plt.close()

        print("A) Mean Strength Comparison:")
        print(f"   Highest Mean Strength: {highest_mean_material} ({means.iloc[0]:.2f} MPa)")
        print(f"   Lowest Mean Strength: {means.index[-1]} ({means.iloc[-1]:.2f} MPa)")

        print("\nB) Variability Comparison (Standard Deviation):")
        print(f"   Highest Variability (Std Dev): {highest_std_material} ({stds.iloc[0]:.2f} MPa)")
        print(f"   Lowest Variability (Std Dev): {stds.index[-1]} ({stds.iloc[-1]:.2f} MPa)")

        # 4. Interpret which material has higher variability
        print("\nC) Variability Interpretation:")
        print(f"The material with the highest variability is **{highest_std_material}**.")
        print("This material's yield strength values are the most spread out from the mean,")
        print("suggesting that its mechanical properties are less consistent and predictable,")
        print("which is a critical factor for engineering design safety margins.")




def plot_distribution_fitting(df_concrete):
    """
    Calculates sample statistics, fits a Normal Distribution to the 'strength_mpa' data,
    compares fitted parameters with sample statistics, and visualizes the fit
    overlaid on a histogram.
    """
    data = df_concrete['strength_mpa']

    sample_mean = data.mean()
    sample_std = data.std()

    # 3. NORMAL DAĞILIMI UYDURMA (FITTING)
    # norm.fit() fonksiyonu, MLE ile en iyi uyan mean (loc) ve std (scale) değerlerini döndürür.
    fitted_mean, fitted_std = norm.fit(data)


    print("--- Compare fitted distribution parameters with sample statistics. ---")
    print("1. SAMPLE STATISTICS:")
    print(f"   Mean : \t{sample_mean:.4f} MPa")
    print(f"   Std Dev : \t{sample_std:.4f} MPa (N-1 Method)")
    print("-" * 50)
    print("2. FITTED DISTRIBUTION PARAMETERS (Uydurulmuş Parametreler):")
    print(f"   Fitted Mean (loc): \t{fitted_mean:.4f} MPa")
    print(f"   Fitted Std Dev (scale): \t{fitted_std:.4f} MPa (MLE Method)")
    print("-" * 50)


    plt.figure(figsize=(10, 6))


    sns.histplot(data, bins=10, color='skyblue', edgecolor='white', alpha=0.7, stat="density", label='Data Histogram')

    # Uydurulmuş Normal Dağılımın PDF'ini Çizme
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    # PDF, fitted mean ve fitted std kullanarak çizilir
    p = norm.pdf(x, fitted_mean, fitted_std)

    plt.plot(x, p, 'r', linewidth=2, label=f'Fitted Normal Curve ($\mu$={fitted_mean:.2f}, $\sigma$={fitted_std:.2f})')

    # Başlık ve Etiketler
    plt.title('Concrete Strength: Histogram with Fitted Normal Distribution', fontsize=14)
    plt.xlabel('Concrete Strength (MPa)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig("Concrete Strength: Histogram with Fitted Normal Distribution.png")
    plt.show()

    real_data = df_concrete['strength_mpa']
    N_SAMPLES = len(real_data)

    # Calculating Fitted Parameters (derived from real data)
    fitted_mean = real_data.mean()
    fitted_std = real_data.std()

    # 2. GENERATE SYNTHETIC DATA
    # Generating new data using the fitted parameters
    synthetic_data = norm.rvs(loc=fitted_mean, scale=fitted_std, size=N_SAMPLES)

    # 3. COMPARE SYNTHETIC VS FITTED PARAMETERS (Validation)
    synth_mean = synthetic_data.mean()
    synth_std = synthetic_data.std()

    print("--- PARAMETER VALIDATION: SYNTHETIC VS FITTED ---")
    print(f"| {'Parameter':<15} | {'Fitted (Target)':<15} | {'Synthetic (Generated)':<15} |")
    print("|" + "-" * 15 + "|" + "-" * 15 + "|" + "-" * 15 + "|")
    print(f"| {'Mean':<15} | {fitted_mean:<15.4f} | {synth_mean:<15.4f} |")
    print(f"| {'Std Dev':<15} | {fitted_std:<15.4f} | {synth_std:<15.4f} |")
    print("-" * 50)
    print(f"Validation: Synthetic parameters closely match the fitted parameters.")

    # 4. VISUAL COMPARISON (Histogram Overlay)
    plt.figure(figsize=(12, 6))

    # Histogram of Original (Real) Data
    sns.histplot(real_data, bins=8, color='skyblue', edgecolor='white', alpha=0.6, stat="density",
                 label='Original (Real) Data')

    # Histogram of Synthetic Data
    sns.histplot(synthetic_data, bins=8, color='orange', edgecolor='white', alpha=0.5, stat="density",
                 label='Synthetic Data')

    # Plot the Fitted Normal Distribution PDF (Reference Curve)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, fitted_mean, fitted_std)
    plt.plot(x, p, 'r', linewidth=2, label='Fitted Normal Curve (Reference)')

    # Titles and Labels
    plt.title('Real vs. Synthetic Data Comparison (Concrete Strength)', fontsize=14)
    plt.xlabel('Concrete Strength (MPa)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig("Real vs. Synthetic Data Comparison (Concrete Strength).png")
    plt.show()



def main():
    """Main execution function."""
    # Load data
    # Perform analyses
    # Generate visualizations
    # Create report

    df_concrete, df_material, df_loads = load_data(concrete_path, material_path, loads_path)
    mean_val,median_val,mode_val, std_val, var_val, skew_val, kurtosis_val = calculate_descriptive_stats(df_concrete)
    plot_material_comparison(df_material)
    prob_exactly_3,prob_at_most_5 = calculate_probability_binomial(100, 0.5)
    prob_exactly_8, prob_more_than_15 = calculate_probability_poisson(10)
    percentage_exceeds_280, percentile_95 = calculate_probability_normal(250,15)
    prob_failure_before_500, prob_survival_beyond_1500 = calculate_probability_exponential(100)
    prob_failure_before_500, prob_survival_beyond_1500 = calculate_probability_exponential(1000)
    apply_bayes_theorem(0.05,0.95,0.90)
    plot_distribution_fitting(df_concrete)

if __name__ == "__main__":
    main()