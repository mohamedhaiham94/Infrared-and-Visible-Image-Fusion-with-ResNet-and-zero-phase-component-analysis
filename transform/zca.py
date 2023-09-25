# credit goes to github.com/mwv
# ------------------------------------
"""zca: ZCA whitening with a sklearn-like interface

"""


from __future__ import division

import numpy as np
from scipy import linalg
import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean, whitening and dewhitening matrices.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        """
        self.whiten_ = torch.zeros_like(X).numpy()
        self.dewhiten_ = torch.zeros_like(X).numpy()
        input_x = X.numpy().copy()
        for i in range(input_x.shape[-1]):
            X = check_array(input_x[:,:,i], accept_sparse=None, copy=self.copy,
                            ensure_2d=True)
            X = as_float_array(X, copy=self.copy)
            self.mean_ = X.mean()
            if self.mean_ > 0:
                X_ = X - self.mean_
                cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
                U, S, _ = linalg.svd(cov)
                s = np.sqrt(S.clip(self.regularization))
                s_inv = np.diag(1./s)
                s = np.diag(s)
                self.whiten_[:, :, i] = np.dot(np.dot(U, s_inv), U.T)
                self.dewhiten_[:, :, i] = np.dot(np.dot(U, s), U.T)
            else:
                self.whiten_[:, :, i] = X

        return self

    def transform(self, X, y=None, copy=None):
        """Perform ZCA whitening

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        output = torch.zeros_like(X).numpy()
        input_x = X.numpy().copy()
        for i in range(output.shape[-1]):
            X = as_float_array(input_x[:, :, i], copy=self.copy)
            output[:, :, i] = np.dot(X - self.mean_, self.whiten_[:, :, i].T)
        
        return output
    
    def inverse_transform(self, X, copy=None):
        """Undo the ZCA transform and rotate back to the original
        representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')

        output = np.zeros_like(X)
        input_x = X.copy()
        for i in range(output.shape[-1]):
            X = as_float_array(input_x[:, :, i], copy=self.copy)
            output[:, :, i] = np.dot(X, self.dewhiten_[:, :, i]) + self.mean_
        
        return output
    
