import numpy as np
import sympy as sp

class LinearTransformationAnalyzer:
    """
    Comprehensive Linear Transformation Analysis Tool
    
    Provides methods to:
    1. Check linearity of a transformation
    2. Compute rank and nullity
    3. Find kernel (null space)
    4. Compute eigenvalues and eigenvectors
    5. Analyze covector space
    """
    
    @staticmethod
    def is_linear_transformation(transformation_func, vector_space_dim):
        """
        Check if a given transformation is linear
        
        Parameters:
        transformation_func (callable): Function representing the transformation
        vector_space_dim (int): Dimension of the input vector space
        
        Returns:
        bool: Whether the transformation is linear
        """
        # Generate random vectors
        np.random.seed(42)
        
        # Check linearity conditions
        for _ in range(10):  # Multiple random tests
            # Generate random vectors
            v1 = np.random.randn(vector_space_dim)
            v2 = np.random.randn(vector_space_dim)
            
            # Scalar for scaling
            c = np.random.randn()
            
            # Linearity tests
            # Test 1: T(v1 + v2) = T(v1) + T(v2)
            additivity_test = np.allclose(
                transformation_func(v1 + v2),
                transformation_func(v1) + transformation_func(v2)
            )
            
            # Test 2: T(c * v1) = c * T(v1)
            homogeneity_test = np.allclose(
                transformation_func(c * v1),
                c * transformation_func(v1)
            )
            
            # If either test fails, it's not a linear transformation
            if not (additivity_test and homogeneity_test):
                return False
        
        return True
    
    @staticmethod
    def matrix_transformation_analysis(matrix):
        """
        Comprehensive analysis of a linear transformation represented by a matrix
        
        Parameters:
        matrix (numpy.ndarray): Matrix representing the linear transformation
        
        Returns:
        dict: Detailed transformation properties
        """
        # Convert to numpy array to ensure compatibility
        A = np.array(matrix, dtype=float)
        
        # Compute rank
        rank = np.linalg.matrix_rank(A)
        
        # Compute nullity (dimension of null space)
        nullity = A.shape[1] - rank
        
        # Compute null space (kernel)
        kernel = sp.Matrix(A).nullspace()
        
        # Eigenvalue and eigenvector computation
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Covector space (dual space)
        covectors = np.linalg.pinv(A).T
        
        return {
            'matrix': A,
            'rank': rank,
            'nullity': nullity,
            'kernel': kernel,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'covectors': covectors
        }
    
    @staticmethod
    def symbolic_transformation_analysis(symbolic_matrix):
        """
        Symbolic analysis of linear transformation
        
        Parameters:
        symbolic_matrix (sympy.Matrix): Symbolic matrix representation
        
        Returns:
        dict: Symbolic transformation properties
        """
        # Compute characteristic polynomial
        x = sp.Symbol('x')
        char_poly = symbolic_matrix.charpoly(x)
        
        # Compute eigenvalues symbolically
        symbolic_eigenvalues = sp.solve(char_poly, x)
        
        # Null space computation
        symbolic_kernel = symbolic_matrix.nullspace()
        
        return {
            'characteristic_polynomial': char_poly,
            'symbolic_eigenvalues': symbolic_eigenvalues,
            'symbolic_kernel': symbolic_kernel
        }

def main():
    # Example linear transformation
    def example_transformation(v):
        """
        Example linear transformation: 
        T([x, y]) = [2x, x+y]
        """
        return np.array([2*v[0], v[0] + v[1]])
    
    # Check linearity
    print("Linear Transformation Linearity Check:")
    is_linear = LinearTransformationAnalyzer.is_linear_transformation(
        example_transformation, 
        vector_space_dim=2
    )
    print("Is Linear:", is_linear)
    
    # Matrix transformation analysis
    print("\nMatrix Transformation Analysis:")
    A = np.array([
        [1, 2],
        [3, 4]
    ])
    analysis = LinearTransformationAnalyzer.matrix_transformation_analysis(A)
    
    print("Rank:", analysis['rank'])
    print("Nullity:", analysis['nullity'])
    print("Eigenvalues:", analysis['eigenvalues'])
    
    # Symbolic analysis
    print("\nSymbolic Transformation Analysis:")
    sym_A = sp.Matrix([
        [sp.Symbol('a'), sp.Symbol('b')],
        [sp.Symbol('c'), sp.Symbol('d')]
    ])
    sym_analysis = LinearTransformationAnalyzer.symbolic_transformation_analysis(sym_A)
    
    print("Symbolic Eigenvalues:", sym_analysis['symbolic_eigenvalues'])

if __name__ == "__main__":
    main()
