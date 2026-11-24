# Principles of AI: Handwritten Digit Recognition

Coursework notebook for classifying handwritten digits (0 to 9).

`main.ipynb` reads `dataset/labels.csv`, loads the 28x28 grayscale images with OpenCV, normalizes the pixel arrays, and fits a multinomial `LogisticRegression` (`lbfgs`, `StandardScaler`) on an 80/20 stratified split. It also prints the confusion matrix, classification report, and checks misclassified samples with their predicted probabilities.

Dependencies: Python 3, NumPy, pandas, OpenCV, scikit-learn.
