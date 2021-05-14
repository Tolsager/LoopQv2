from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

from utils import *

seed_everything(SEED)

# load data
df_train = pd.read_csv(CSV_TRAIN)
n_train = len(df_train)

y = df_train['emotion'].values
X = np.zeros((n_train, 48*48))

print("Reading images")
for i in tqdm.tqdm(range(n_train)):
    image_path = os.path.join(IMAGE_DIRECTORY_TRAIN, df_train['image_id'].iloc[i] + '.jpg')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_flat = image.flatten()
    X[i, :] = image_flat

# standardize the data
scl = StandardScaler()
X = scl.fit_transform(X)

# project the images onto the 104 largest principal components to capture 90% variance
pca = PCA(n_components=104)
X = pca.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=y)

# setup classifiers
lr = LogisticRegression(max_iter=1000)
gbc = GradientBoostingClassifier(n_estimators=50, min_samples_split=4, verbose=1)
svm = LinearSVC(max_iter=2000, verbose=1)

clf_names = ["Logistic regression", "Gradient boosted trees", "Support vector machine"]
clfs = [lr, gbc, svm]


def test_clf(clf):
    clf.fit(X_train, y_train)
    accuracy_train = clf.score(X_train, y_train)
    accuracy_test = clf.score(X_test, y_test)
    return accuracy_train, accuracy_test


print()
print("Training models")
accuracies_train = []
accuracies_test = []
for clf, name in zip(clfs, clf_names):
    print(f"Training model: {name}")
    print()
    accuracy_train, accuracy_test = test_clf(clf)
    accuracies_train.append(accuracy_train)
    accuracies_test.append(accuracy_test)


print("Comparing models")
for i in range(len(clfs)):
    print(f"Model: {clf_names[i]}")
    print(f"    Training accuracy: {accuracies_train[i]}")
    print(f"    Test accuracy: {accuracies_test[i]}")
    print()
