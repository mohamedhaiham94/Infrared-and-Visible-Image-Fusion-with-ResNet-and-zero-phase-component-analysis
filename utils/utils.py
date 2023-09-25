import numpy as np

def l1_norm(features, unit=5):
  """Computes the L1-norm of a 3D array of feature maps.

  Args:
    features: A 3D numpy array of feature maps.
    unit: The size of the neighborhood used to compute the L1-norm.

  Returns:
    A 2D numpy array of L1-norm values.
  """

  # Pad the features array.
  padded_features = np.pad(features, ((unit // 2, unit // 2), (unit // 2, unit // 2), (0, 0)), 'constant', constant_values=0)

  # Compute the L1-norm of each pixel.
  l1_norm_values = np.zeros((features.shape[0], features.shape[1]))
  for i in range(unit // 2, features.shape[0] + unit // 2):
    for j in range(unit // 2, features.shape[1] + unit // 2):
      l1_norm_values[i - unit // 2, j - unit // 2] = np.sum(np.abs(padded_features[i - unit // 2:i + unit // 2 + 1, j - unit // 2:j + unit // 2 + 1, :])) / (unit * unit)

  return l1_norm_values