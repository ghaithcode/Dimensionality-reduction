# Reducing Features Using Principle Components


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()

# Standardize the feature matrix 
features = StandardScaler().fit_transform(digits.data)

pca = PCA(n_components=0.99, whiten=True)
features_pca = pca.fit_transform(features)

# show results of the features reduction 
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])
