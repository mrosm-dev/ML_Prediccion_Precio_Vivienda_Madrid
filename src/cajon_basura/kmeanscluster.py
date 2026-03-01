from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeansCluster(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X_scaled)
        return self
    
    def transform(self, X):
        X_scaled = self.scaler_.transform(X)
        labels = self.kmeans_.predict(X_scaled)
        letters = [chr(ord('A') + l) for l in labels]
        X['cluster'] = letters
        return X
    
    def get_feature_names_out(self, input_features=None):
        return list(input_features) + ['cluster']