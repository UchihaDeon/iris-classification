from src.preprocessing import load_data, prepare_features_labels
from src.model import train_knn, train_decision_tree
from src.evaluate import evaluate_model

# Load and prepare data
df = load_data("data/iris.csv")
X, y = prepare_features_labels(df)

# Train and evaluate KNN
print("=== KNN Model ===")
knn_model, X_test, y_test = train_knn(X, y, k=5)
evaluate_model(knn_model, X_test, y_test)

# Train and evaluate Decision Tree
print("\n=== Decision Tree Model ===")
dt_model, X_test, y_test = train_decision_tree(X, y)
evaluate_model(dt_model, X_test, y_test)