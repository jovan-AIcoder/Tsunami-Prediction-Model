from tensorflow import keras
import pandas as pd
import joblib
import sklearn.preprocessing as pp
df = pd.read_csv('tsunami.csv')
X = df.drop('tsunami', axis=1)
y = df['tsunami']
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)
model = keras.models.load_model('tsunami_model.h5')
#loss and accuracy evaluation
loss, accuracy = model.evaluate(X_scaled, y)
print(f'Model Loss: {loss}')
print(f'Model Accuracy: {accuracy}')
#predictions
y_pred = model.predict(X_scaled)
y_pred_classes = (y_pred > 0.5).astype(int)
#precision, recall, f1-score
from sklearn.metrics import classification_report
print(classification_report(y, y_pred_classes))
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred_classes)
print('Confusion Matrix:')
print(cm)
#ROC curve and AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',
            lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()