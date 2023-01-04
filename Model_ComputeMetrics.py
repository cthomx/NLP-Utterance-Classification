import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# evaluate the training accuracy by predicted result and true label y_train
def getEvaluationMetrics(trueLabel, predictedLabel, classifierName, dataSetType):
    print(dataSetType + ' Accuracy:', np.mean(trueLabel == predictedLabel))
    f1_score_vector = f1_score(trueLabel, predictedLabel, average=None)
    print('F1 score :', np.mean(trueLabel == predictedLabel))
    print('f1 score using ' + classifierName + 'classifier is :', np.mean(f1_score_vector))
    print('Confusion matrix :\n', confusion_matrix(trueLabel, predictedLabel))
    # regular confusion matrix
    fig, ax = plt.subplots() 
    sns.heatmap(confusion_matrix(trueLabel, predictedLabel), annot=True, ax=ax, fmt='g')
    plt.show()