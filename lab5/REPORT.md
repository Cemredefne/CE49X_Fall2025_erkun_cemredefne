# Lab 5 Report: Bias–Variance

Fill in concise answers under each heading. You can paste plots or reference notebook cells.

## Question 1: Data and baseline model
1) The polynomial degree that provides the best generalization is Degree 1 (Simple Linear Regression). This conclusion is based on the Cross-Validation (CV) analysis, which is statistically the most reliable method. The CV analysis showed that Degree 1 had the lowest average RMSE (approx 1.4733), indicating the most stable and reliable performance across different test sets. This result directly contrasts with your Single Train/Test Split analysis, which found the minimum test error (approx 1.4101) at Degree 7. The single split result, while lower in error, is less reliable because the CV process proved that models of Degree 2 and higher are highly unstable and prone to High Variance when tested on multiple data partitions. The lower single-split RMSE value is merely an optimistic outcome due to a favorable random data split.

## Question 2: Model complexity vs error
2) As the polynomial degree increases, the errors exhibit the standard Bias-Variance phases. Initially, at low complexity (Degree 1), both Training RMSE and Testing RMSE are relatively high, marking the High Bias (Underfitting) zone. As complexity increases, the Training RMSE consistently decreases (the model fits better). The Testing RMSE generally decreases until the Optimal Spot is reached (d=1 in CV, or d=7 in the single split). Crucially, beyond this optimal point, in the High Complexity (Overfitting) region (Degrees 8–10), the Training RMSE continues its slight decline (model memorizing noise), while the Testing RMSE either becomes highly unstable or increases sharply (as confirmed by the large error spikes in the full CV analysis), demonstrating the loss of generalization ability.

## Question 3: Bias vs Variance interpretation
3) Bias (Underfitting) manifests at low complexity (Degree 1). Here, the simple model lacks the flexibility to accurately map the non-linear relationship between the features (T, RH, AH) and the target (CO(GT)), resulting in systematic errors. Variance (Overfitting) manifests at high complexity (Degrees 7–10). This is clearly shown by the significant gap between the low Training RMSE (approx 1.39) and the unstable Testing RMSE observed in your results. The model incorporates over 200 polynomial terms (at d=10) to memorize noise in the training set. The dramatic increase in the CV average RMSE for d > 1 proves that this High Variance is real, causing the complex model to fail reliably on unseen data.

## Question 4: Regularization effect
4) Sensor Noise primarily exacerbates Variance. If a complex model (e.g., Degree 7) is trained on noisy data, it will treat the random measurement errors as real predictive patterns. This reliance on noise causes the model's predictions to be overly sensitive and erratic on clean test data, resulting in dramatically high variance and poor generalization. Missing Data  affects Bias via data quality. Although we used the targeted cleaning method, if the data removal was non-random (e.g., removing only high-temperature readings), the remaining dataset becomes non-representative. This skewed training data forces the model to learn a structurally biased relationship, potentially leading to errors that cannot be solved by simply increasing model complexity.

## Question 5: Conclusions and recommendations
- Final takeaway(s) about model selection for this dataset:
- Recommended configuration(s) and why:
- Any caveats or next steps:


