from sklearn.ensemble import RandomForestClassifier

print("Training production model...")

print("Starting Random Forest experiment...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Model:", model)