# 0. Imports 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset from CSV file into a Pandas DataFrame
df = pd.read_csv("women_safety.csv")

# 2. Quick look at data: show first few rows
print("First 30 rows of the dataset:")
print((df.head(30)), "\n")

# 3. (Optional) Check for missing values
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# 4. Encode categorical columns separately
le_area = LabelEncoder()
le_time = LabelEncoder()

df["AreaTypeEncoded"] = le_area.fit_transform(df["AreaType"])
df["TimeOfDayEncoded"] = le_time.fit_transform(df["TimeOfDay"])

# 5. Prepare features (X) and target (y)
feature_cols = ["AreaTypeEncoded", "TimeOfDayEncoded", "PolicePatrol", "LightLevel", "CrowdDensity", "CrimeRate"]
X = df[feature_cols]
y = df["Safety"]  # target: 1 = Safe, 0 = Unsafe

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Create and train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate model on the test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Get feature importances (which inputs matter most)
importances = model.feature_importances_
for col, imp in zip(feature_cols, importances):
    print(f"{col}: {imp:.3f}")

# 10. Safe prediction function (robust to class ordering)
# Updated safe prediction function
def predict_safety(area, time, police, light, crowd, crime):
    """
    Input values:
      - area: string e.g. "Market", "Park", "Residential", "Highway", "OfficeArea", "BusStop"
      - time: string e.g. "Morning", "Evening", "Night"
      - police: int (0-2)
      - light: int (0-2)
      - crowd: int (0-5)
      - crime: int (0-5)
    Returns:
      - None (prints prediction and probability)
    """
    try:
        # Encode categorical inputs using fitted LabelEncoders
        area_enc = le_area.transform([area])[0]
        time_enc = le_time.transform([time])[0]

        # Create a DataFrame with same column names used in training
        input_df = pd.DataFrame(
            [[area_enc, time_enc, police, light, crowd, crime]],
            columns=feature_cols
        )

        # Get prediction and probability
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Get probability of 'Safe' class
        try:
            idx_safe = list(model.classes_).index(1)
            safety_prob = proba[idx_safe] * 100.0
        except ValueError:
            safety_prob = proba.max() * 100.0

        # Display results
        status = "✅ SAFE" if pred == 1 else "⚠️ UNSAFE"
        print(f"Predicted Status: {status}")
        print(f"Safety Score: {safety_prob:.1f}%")
        
    except ValueError as e:
        if "previously unseen labels" in str(e):
            print(f"Error: Unknown area type '{area}' or time '{time}'")
            print(f"Available area types: {list(le_area.classes_)}")
            print(f"Available time periods: {list(le_time.classes_)}")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# 11. Example usage
print("\nExample predictions:")
predict_safety("Market", "Night", 1, 2, 2, 0)     # expected safer example
"""(areatype,timeofday,policepatrol,LightLevel,crowddensity,crimerate)"""