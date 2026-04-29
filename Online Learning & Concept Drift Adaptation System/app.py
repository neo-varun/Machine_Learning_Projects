import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from river.drift import ADWIN


def data_stream(n_samples=10000, drift_point=2000):
    for i in range(n_samples):
        if i < drift_point:
            X = np.random.normal(loc=0.0, scale=1.0, size=2)
            y = 0 if X[0] + X[1] < 0 else 1
        else:
            X = np.random.normal(loc=2.0, scale=1.5, size=2)
            y = 0 if X[0] - X[1] < 0 else 1
        yield X, y


model = SGDClassifier(loss="log_loss")
classes = np.array([0, 1])
drift_detector = ADWIN()

accuracies = []
preds = []
true_labels = []
drift_points = []

window_preds = []
window_true = []

for i, (X, y) in enumerate(data_stream()):
    X = X.reshape(1, -1)

    if i == 0:
        y_pred = 0
    else:
        y_pred = model.predict(X)[0]

    preds.append(y_pred)
    true_labels.append(y)

    error = int(y_pred != y)
    drift_detector.update(error)

    window_preds.append(y_pred)
    window_true.append(y)

    if i == 0:
        model.partial_fit(X, [y], classes=classes)
    else:
        model.partial_fit(X, [y])

    if drift_detector.drift_detected:
        print(f"Drift detected at index {i}")
        drift_points.append(i)
        model = SGDClassifier(loss="log_loss")
        model.partial_fit(X, [y], classes=classes)
        window_preds = []
        window_true = []

    if len(window_true) > 50:
        acc = accuracy_score(window_true, window_preds)
        accuracies.append(acc)

drift_point = 2000

pre_drift_acc = accuracy_score(true_labels[:drift_point], preds[:drift_point])
post_drift_acc = accuracy_score(true_labels[drift_point:], preds[drift_point:])

print("Accuracy BEFORE drift:", round(pre_drift_acc, 4))
print("Accuracy AFTER drift :", round(post_drift_acc, 4))
print("Accuracy variance    :", round(np.std(accuracies), 4))

plt.figure()
plt.plot(accuracies)
plt.axvline(x=drift_point / 50, linestyle="--", label="True Drift Point")
for dp in drift_points:
    plt.axvline(x=dp / 50, linestyle=":", label="Detected Drift")
plt.title("Accuracy Over Time (Streaming Model)")
plt.xlabel("Time (windowed)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
