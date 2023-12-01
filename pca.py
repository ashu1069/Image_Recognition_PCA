def PCA(x, N):
  '''
  Arguments:
  x: input features
  N: number fo components
  '''
    #mean
    x_mean = np.mean(x)
    
    #standard deviation
    x_std = np.std(x)
    
    #normalized values
    z = (x - x_mean)/x_std
    
    #covariance matrix
    cov_matrix = np.cov(z)
    
    #eigen values and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    
    #index the eigenvalues in descending order
    sorted_idx = eigen_values.argsort()[::-1]
    
    sorted_eigenvalues = eigen_values[sorted_idx]
    
    sorted_eigenvectors = eigen_vectors[:,sorted_idx]
    
    #selecting the first n eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:N]
    
    #data transformation
    x_transformed = np.dot(eigenvector_subset.transpose(), x_mean.transpose()).transpose()
    
    return x_transformed
