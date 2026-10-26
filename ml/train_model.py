import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('../dataset/fertilizer_recommendation_dataset.csv')

X = df[['Temperature', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'PH', 'Soil', 'Crop']]
y = df['Fertilizer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

categorical_features = ['Soil', 'Crop']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=2, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracy:.2f}%")

joblib.dump(model, 'saved_model.pkl')