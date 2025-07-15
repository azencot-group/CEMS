import torch
import numpy as np
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_np = None
        self.max_np = None
        self.min_torch = None
        self.max_torch = None

    def fit(self, X):
        """Compute the minimum and maximum to be used for scaling."""
        if isinstance(X, torch.Tensor):
            self.min_torch = X.min(dim=0, keepdim=True)[0]
            self.max_torch = X.max(dim=0, keepdim=True)[0]
            self.min_np = self.min_torch.cpu().numpy()
            self.max_np = self.max_torch.cpu().numpy()
        elif isinstance(X, np.ndarray):
            self.min_np = np.min(X, axis=0, keepdims=True)
            self.max_np = np.max(X, axis=0, keepdims=True)
            self.min_torch = torch.from_numpy(self.min_np).to(device='cuda',dtype=torch.float32)
            self.max_torch = torch.from_numpy(self.max_np).to(device='cuda',dtype=torch.float32)

    def transform(self, X):
        """Scale features of X according to feature_range."""
        if isinstance(X, torch.Tensor):
            range_ = self.max_torch - self.min_torch
            scale = (self.feature_range[1] - self.feature_range[0])/ range_
            return self.feature_range[0] + (X - self.min_torch.to(X.device)) * scale.to(X.device)
        elif isinstance(X, np.ndarray):
            range_ = self.max_np - self.min_np
            scale = (self.feature_range[1] - self.feature_range[0]) / range_
            return self.feature_range[0]+ (X - self.min_np) * scale
        else:
            raise TypeError("Input must be a numpy array or a torch tensor")



    def inverse_transform(self, X_scaled):
        """Reverse the scaling to original feature values."""
        if isinstance(X_scaled, torch.Tensor):
            range_ = self.max_torch - self.min_torch
            inverse = self.min_torch.to(X_scaled.device) + (X_scaled - self.feature_range[0]) * (range_.to(X_scaled.device) / (self.feature_range[1] - self.feature_range[0]))
        else:
            range_ = self.max_np - self.min_np
            inverse = self.min_np + (X_scaled - self.feature_range[0]) * (range_ / (self.feature_range[1] - self.feature_range[0]))
        return inverse

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
