class AlignClouds:
    """
    A class for estimating the affine transform parameters that minimize the ICP.
    between two 3D point clouds.
    """
    
    def __init__(self, ex_2):
        """
        Initializes the PointCloudAffineAlignment object.
        """
        self.tree = KDTree(ex_2[:,:3])

    def loss_function(self, A, B):
        """
        Compute the alignment between two sets of points.

        Parameters:
        - A: np.ndarray, the source point cloud as an Nx3 numpy array.
        - B: np.ndarray, the target point cloud as an Mx3 numpy array.

        Returns:
        - float, the Chamfer distance between point cloud A and B.
        """
        query_responses = self.tree.query(A[:,:3])
        return np.sum((A[:,:3] - B[query_responses[1],:3]) ** 2)

    @staticmethod
    def transform_points(points, params):
        """
        Apply an affine transformation to the points.

        Parameters:
        - points: np.ndarray, the point cloud to transform as an Nx3 numpy array.
        - params: np.ndarray, the transformation parameters (12 in total: 9 for the top-left part
                  representing rotation, scaling, and shearing, followed by 3 for translation).

        Returns:
        - np.ndarray, the transformed point cloud as an Nx3 numpy array.
        """
        # assert len(params) == 12, "Affine transformation parameters must be of length 12."
        affine_matrix = np.reshape(params[:9], (3, 3))
        translation_vector = params[-3:]
        transformed_points = points.copy()
        transformed_points[:,:3] = np.dot(points[:,:3], affine_matrix.T) + translation_vector
        return transformed_points

    def objective_function(self, params, source, target):
        """
        The objective function to minimize during optimization.

        Parameters:
        - params: np.ndarray, the current estimate of the affine transformation parameters.
        - source: np.ndarray, the source point cloud as an Nx3 numpy array.
        - target: np.ndarray, the target point cloud as an Mx3 numpy array.

        Returns:
        - float, the current Chamfer distance between the transformed source and target point clouds.
        """
        transformed_source = self.transform_points(source, params)
        d = self.loss_function(transformed_source, target)
        return d

    def estimate_transform(self, source, target):
        """
        Estimate the affine transform parameters that minimize the Chamfer distance
        between the source and target point clouds.

        Parameters:
        - source: np.ndarray, the source point cloud as an Nx3 numpy array.
        - target: np.ndarray, the target point cloud as an Mx3 numpy array.

        Returns:
        - np.ndarray, the optimal affine transformation parameters.
        """
        # Initial guess (9 for affine part + 3 for translation, here initialized to identity + zero translation)
        initial_params = np.concatenate((np.eye(3).reshape((-1,)), np.mean(target[:,:3],axis=0)-np.mean(source[:,:3],axis=0)))
        result = minimize(self.objective_function, initial_params, args=(source, target), 
                          method='L-BFGS-B', options={'disp': True})
        # result = basinhopping(self.objective_function, initial_params, minimizer_kwargs={'args': (source, target)}, disp=True)
        self.optimal_params = result.x
        return result.x