import joblib

# Load the trained models from files
model_y1_filename = "linear_model_y1_elnet.joblib"
model_y2_filename = "linear_model_y2_elnet.joblib"

linear_model_y1 = joblib.load(model_y1_filename)
linear_model_y2 = joblib.load(model_y2_filename)

# Access the coefficients and feature names through the underlying model
coefficients_y1 = linear_model_y1.model.coef_
features_y1 = linear_model_y1.model.feature_names_in_

coefficients_y2 = linear_model_y2.model.coef_
features_y2 = linear_model_y2.model.feature_names_in_

# Print or use the coefficients and features as needed
print("Linear Model Y1 Coefficients:")
for feature, coef in zip(features_y1, coefficients_y1):
    print(f"{feature}: {coef}")

print("\nLinear Model Y2 Coefficients:")
for feature, coef in zip(features_y2, coefficients_y2):
    print(f"{feature}: {coef}")