import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.special import comb
from sklearn.model_selection import cross_val_score

air_path = "/Users/student/PycharmProjects/labs/datasets/AirQualityUCI.csv"

#Data loading
#Step 1.1 - Load the Data
def load_data(air_path):
    try:
        #Data columns are separated with ";". The sep = ";" comment specifies this to Pandas.
        #In the data, pandas reads the time column as object/string.
        #The comma is a decimal separator in the data, and is converted to float with decimal = ",".
        df_air = pd.read_csv(air_path, sep=';', decimal=',')
        #Converting the columns to strings, and cleaning the noise.
        df_air.columns = df_air.columns.str.strip()
        initial_rows = len(df_air)
        #Replacing (-200) to NaN
        #inplace=True, changes the current dataframe.
        df_air.replace(to_replace=-200, value=np.nan, inplace=True)
        #There is empty columns in the data. Axis=1 refers to the columns, Axis=0 refers to the rows. How="all" specifies that only columns that are completely empty will be dropped.
        df_air.dropna(axis=1, how='all', inplace=True)
        subset_cols = ['T', 'RH', 'AH', 'CO(GT)']
        df_air.dropna(subset=subset_cols, inplace=True)
        dropped_rows = initial_rows - len(df_air)
        print("--- Missing Value Handling ---")
        print(f"Air Quality UCL Dataset: Initial rows: {initial_rows}, Dropped rows: {dropped_rows}, (Remaining: {len(df_air)} rows)\n")
        return df_air
    except FileNotFoundError:
        print("Error: One or more dataset files not found.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

#Column Selecting
#Step 1.2 — Prepare the Data
def column_selecting(df_air):
    features = ['T', 'RH', 'AH']
    target = 'CO(GT)'
    features = df_air[features]
    target = df_air[target]
    #test_size = 0.3 splits test into training (70%) and testing (30%).
    #random_state = 42
    #The random_state = 42 parameter sets a seed for the random number generator.
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)
    print("--- Train/Test Split Dimensions ---")
    print(f"Training Features (features_train) Shape: {features_train.shape}")
    print(f"Testing Features (features_test) Shape: {features_test.shape}")
    print(f"Training Target (target_train) Shape: {target_train.shape}")
    print(f"Testing Testing (target_test) Shape: {target_test.shape}")
    print("--- Splitting Complete ---")
    return features_train, features_test, target_train, target_test

#Step 2 — Fit Models of Increasing Complexity
def linear_reg(features_train, features_test, target_train, target_test):
    train_errors = []
    test_errors = []
    degrees = range(1, 11)

    for d in degrees:
        poly_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('lin_reg', LinearRegression())
        ])
        poly_model.fit(features_train, target_train)
        target_train_pred = poly_model.predict(features_train)
        target_test_pred = poly_model.predict(features_test)
        train_rmse = np.sqrt(mean_squared_error(target_train, target_train_pred))
        test_rmse = np.sqrt(mean_squared_error(target_test, target_test_pred))
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

        print(f"Model Degree {d:2}: Training RMSE = {train_rmse:.4f}, Testing RMSE = {test_rmse:.4f}")
    print("--- All 10 models have been trained successfully. ---")
    return degrees, train_errors, test_errors

#Step 3 — Plot the Validation Curve
def Validation_Curve(degrees, train_errors, test_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, label='Training RMSE', marker='o')
    plt.plot(degrees, test_errors, label='Testing RMSE', marker='o')
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Bias–Variance Tradeoff')
    plt.xticks(list(degrees))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    min_test_error_index = test_errors.index(min(test_errors))
    optimal_degree = list(degrees)[min_test_error_index]
    min_error = min(test_errors)
    plt.axvline(x=optimal_degree, color='g', linestyle='--', linewidth=1.5, label='Optimal Complexity')
    plt.annotate(f'Optimal Degree: {optimal_degree}',
                 xy=(optimal_degree, min_error),
                 xytext=(optimal_degree + 1, min_error * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=0.5),
                 fontsize=10)
    plt.text(1.5, max(test_errors) * 0.9, 'UNDERFITTING\n(High Bias)', color='red', fontsize=12,
             horizontalalignment='center')
    plt.text(8.5, max(test_errors) * 0.9, 'OVERFITTING\n(High Variance)', color='blue', fontsize=12,
             horizontalalignment='center')
    plt.savefig("validation_curve.png")

    plt.show()

#Extra 4.1 - Plotting Graphs

def generate_analysis_plots(degrees, train_errors, test_errors, n_features_original=3):
    feature_counts = [int(comb(n_features_original + d, d) - 1) for d in degrees]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, feature_counts, label='Total Polynomial Features', marker='o', color='purple')
    plt.title('1. Model Complexity Growth (Feature Count vs. Degree)')
    plt.xlabel('Polynomial Degree (d)')
    plt.ylabel('Number of Features Created (Log Scale)')
    plt.xticks(list(degrees))
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.savefig("complexity_growth_analysis.png")
    plt.show()

    generalization_gap = np.array(test_errors) - np.array(train_errors)
    optimal_degree = degrees[np.argmin(test_errors)]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, generalization_gap, label='Testing RMSE - Training RMSE', marker='o', color='red')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(optimal_degree, color='g', linestyle='--', label=f'Optimal Degree: {optimal_degree}')
    plt.title('2. Overfitting Indicator: Generalization Gap')
    plt.xlabel('Polynomial Degree (d)')
    plt.ylabel('Generalization Gap (Variance)')
    plt.xticks(list(degrees))
    plt.legend()
    plt.grid(True)
    plt.savefig("overfitting_indicator_analysis.png")
    plt.show()

#Extra 4.2 - Plotting Graphs
def plot_residual_analysis(optimal_degree, features_train, target_train, features_test, target_test):
    optimal_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=optimal_degree, include_bias=False)),
        ('lin_reg', LinearRegression())
    ])

    optimal_model.fit(features_train, target_train)

    y_test_pred = optimal_model.predict(features_test)
    residuals = target_test - y_test_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), color='red', linestyle='--')
    plt.title(f'3. Residual Plot for Optimal Model (Degree {optimal_degree})')
    plt.xlabel('Predicted CO(GT) Value')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.savefig("residual_plot_analysis.png")
    plt.show()


#Extra 5.1 - Cross Validation
def cross_val_analysis(df_air):
    X = df_air[['T', 'RH', 'AH']]
    y = df_air['CO(GT)']
    degrees = range(1, 11)
    cv_rmses = []
    print()
    print("--- Cross-Validation Analysis (k=5) ---")
    for d in degrees:
        # Pipeline oluşturulur
        poly_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('lin_reg', LinearRegression())
        ])
        scores = cross_val_score(
            poly_model,
            X,
            y,
            scoring='neg_mean_squared_error',
            cv=5
        )
        avg_rmse = np.mean(np.sqrt(-scores))
        cv_rmses.append(avg_rmse)
        print(f"Model Degree {d:2}: Average CV RMSE = {avg_rmse:.4f}")

    optimal_degree_cv = degrees[np.argmin(cv_rmses)]
    min_cv_rmse = np.min(cv_rmses)

    print("-" * 40)
    print(f"Optimal Degree (CV): {optimal_degree_cv} (Min CV RMSE: {min_cv_rmse:.4f})")
    print("-" * 40)
    return degrees, cv_rmses

#Extra 5.2 - Cross Validation Plot
def plot_cv_validation_curve(degrees, cv_rmses):
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, cv_rmses, label='Average CV RMSE (k=5)', marker='o', color='darkorange')

    optimal_degree_cv = degrees[np.argmin(cv_rmses)]
    min_error = np.min(cv_rmses)

    plt.axvline(x=optimal_degree_cv, color='red', linestyle='--', linewidth=1.5,
                label=f'CV Optimal Degree: {optimal_degree_cv}')

    plt.title('Validation Curve: Cross-Validation Error')
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Average Root Mean Squared Error (RMSE)')
    plt.xticks(list(degrees))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("cv_validation_curve.png")
    plt.show()

#Extra 5.4 - Comparison
def compare(optimal_degree,optimal_degree_cv):
    print("\n=== FINAL MODEL COMPARISON ===")
    print(f"1. Single Split Optimal Degree: {optimal_degree}")
    print(f"2. Cross-Validation Optimal Degree: {optimal_degree_cv}")
    print("\nCOMMENT:")
    if optimal_degree_cv < optimal_degree:
        print(f"Cross-Validation (CV) showed a significant shift toward simplicity (Degree {optimal_degree_cv}).")
        print("This indicates that the single split result (Degree {optimal_degree}) was overly optimistic.")
        print(
            "CV proves that higher-degree models are unstable and highly prone to High Variance/Overfitting when tested on different data partitions.")
    elif optimal_degree_cv == optimal_degree:
        print(
            "Both analysis methods agree on the optimal complexity. This indicates the model is highly stable across different data splits.")
    else:
        print(f"CV suggests a slightly more complex model (Degree {optimal_degree_cv}) is optimal.")
        print(
            "This means the single split analysis was pessimistic, and CV found a better average fit through wider testing.")

    print("==============================")

def main():
    df_air = load_data(air_path)
    features_train, features_test, target_train, target_test = column_selecting(df_air)
    degrees, train_errors, test_errors = linear_reg(features_train, features_test, target_train, target_test)
    test_errors_array = np.array(test_errors)
    optimal_index = np.argmin(test_errors_array)
    optimal_degree = degrees[optimal_index]
    print(f"\nFINAL SPLIT ANALYSIS: Optimal Degree is {optimal_degree} (Min Test RMSE: {test_errors[optimal_index]:.4f})")
    Validation_Curve(degrees, train_errors, test_errors)
    generate_analysis_plots(degrees, train_errors, test_errors)
    plot_residual_analysis(
        optimal_degree,
        features_train,
        target_train,
        features_test,
        target_test
    )
    degrees_cv, cv_rmses = cross_val_analysis(df_air)
    optimal_degree_cv = degrees_cv[np.argmin(cv_rmses)]
    plot_cv_validation_curve(degrees_cv, cv_rmses)
    compare(optimal_degree,optimal_degree_cv)
    

if __name__ == "__main__":
    main()
