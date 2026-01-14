import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import  load_model

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = load_model("best_model.h5")

y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis = 1)
y_true_classes = np.argmax(y_test, axis = 1)

activity_labels = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"
]

print(classification_report(y_true_classes,y_pred_classes, target_names= activity_labels))

cm = confusion_matrix(y_pred_classes, y_true_classes)

plt.figure()
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels= activity_labels, yticklabels=activity_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predict")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()