import numpy as np
import sympy as sp
import itertools

import numpy as np
import numpy.linalg as la

class NumPyLinearAlgebraDemo:
    """
    Comprehensive demonstration of NumPy's linear algebra capabilities
    """
    
    @staticmethod
    def matrix_operations():
        """
        Demonstrate key linear algebra operations in NumPy
        """
        print("NumPy Linear Algebra Module (numpy.linalg) Demonstration:\n")
        
        # Create sample matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # 1. Matrix Rank
        print("1. Matrix Rank:")
        print("Rank of A:", la.matrix_rank(A))
        
        # 2. Determinant
        print("\n2. Determinant:")
        print("Det(A):", la.det(A))
        
        # 3. Matrix Inverse
        print("\n3. Matrix Inverse:")
        try:
            A_inv = la.inv(A)
            print("Inverse of A:\n", A_inv)
            # Verify inverse
            print("\nVerification (A * A_inv should be identity):\n", 
                  np.round(A @ A_inv, decimals=10))
        except la.LinAlgError as e:
            print("Matrix is not invertible:", e)
        
        # 4. Eigenvalues and Eigenvectors
        print("\n4. Eigenvalues and Eigenvectors:")
        eigenvalues, eigenvectors = la.eig(A)
        print("Eigenvalues:", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
        
        # 5. Singular Value Decomposition (SVD)
        print("\n5. Singular Value Decomposition:")
        U, S, Vt = la.svd(A)
        print("U (left singular vectors):\n", U)
        print("Singular values:", S)
        print("V^T (right singular vectors transposed):\n", Vt)
        
        # 6. Solving Linear Equations
        print("\n6. Solving Linear Equations (Ax = b):")
        b = np.array([5, 11])
        x = la.solve(A, b)
        print("Solution x:", x)
        print("Verification (A * x == b):", np.allclose(A @ x, b))
        
        # 7. Norm Calculations
        print("\n7. Matrix and Vector Norms:")
        print("Vector 2-norm:", la.norm([3, 4]))  # Euclidean norm
        print("Matrix Frobenius norm:", la.norm(A, 'fro'))
    
    @staticmethod
    def advanced_linear_algebra():
        """
        Advanced linear algebra techniques
        """
        print("\nAdvanced Linear Algebra Techniques:\n")
        
        # Pseudo-inverse (Moore-Penrose inverse)
        A = np.array([[1, 2], [3, 4], [5, 6]])
        pinv_A = la.pinv(A)
        print("Pseudo-inverse:\n", pinv_A)
        
        # Condition number
        print("\nCondition Number:", la.cond(A))
        
        # Matrix exponential
        exp_A = la.matrix_power(A, 3)
        print("\nMatrix Power (A^3):\n", exp_A)

def main():
    # Run demonstrations
    NumPyLinearAlgebraDemo.matrix_operations()
    NumPyLinearAlgebraDemo.advanced_linear_algebra()

if __name__ == "__main__":
    main()

class MatrixOperations:
    """
    Comprehensive matrix operations focusing on determinant and permanent
    """
    
    @staticmethod
    def determinant(matrix):
        """
        Compute determinant using different methods
        
        Parameters:
        matrix (numpy.ndarray or list): Input matrix
        
        Returns:
        float: Determinant of the matrix
        """
        # Convert to numpy array
        A = np.array(matrix, dtype=float)
        
        # Method 1: NumPy determinant
        numpy_det = np.linalg.det(A)
        
        # Method 2: Symbolic determinant using SymPy
        sym_matrix = sp.Matrix(matrix)
        symbolic_det = sym_matrix.det()
        
        # Method 3: Manual calculation (for small matrices)
        def manual_det(mat):
            """Recursive determinant calculation"""
            mat = np.array(mat)
            # Base cases
            if mat.shape == (1, 1):
                return mat[0, 0]
            if mat.shape == (2, 2):
                return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
            
            # Recursive case
            det = 0
            for c in range(mat.shape[1]):
                # Create submatrix
                submat = np.delete(np.delete(mat, 0, axis=0), c, axis=1)
                # Recursive calculation with alternating sign
                det += ((-1) ** c) * mat[0, c] * manual_det(submat)
            return det
        
        manual_det_result = manual_det(A)
        
        return {
            'numpy_det': numpy_det,
            'symbolic_det': symbolic_det,
            'manual_det': manual_det_result
        }
    
    @staticmethod
    def permanent(matrix):
        """
        Compute permanent of a matrix
        
        Permanent is similar to determinant but without sign changes
        
        Parameters:
        matrix (numpy.ndarray or list): Input matrix
        
        Returns:
        float: Permanent of the matrix
        """
        def compute_permanent(mat):
            """
            Compute permanent using combinatorial method
            
            Time complexity: O(n! * n)
            """
            mat = np.array(mat)
            n = mat.shape[0]
            perm = 0
            
            # Iterate through all permutations
            for p in itertools.permutations(range(n)):
                # Multiply diagonal elements
                term = 1
                for i in range(n):
                    term *= mat[i, p[i]]
                perm += term
            
            return perm
        
        return compute_permanent(matrix)


class CovarianceContravariance:
    """
    Exploration of Covariance and Contravariance in different domains
    """
    
    @staticmethod
    def mathematical_vectors():
        """
        Covariance and Contravariance in Vector Spaces
        
        In coordinate transformations, vectors transform differently
        """
        print("Mathematical Vector Covariance/Contravariance:")
        
        # Coordinate transformation example
        def coordinate_transform(vector, transform_matrix):
            """
            Show how vectors transform under linear transformation
            
            Covariant vector: v' = A * v
            Contravariant vector: v' = A^(-1) * v
            """
            # Covariant transformation
            covariant_vector = transform_matrix @ vector
            
            # Contravariant transformation
            try:
                contravariant_vector = np.linalg.inv(transform_matrix) @ vector
            except np.linalg.LinAlgError:
                contravariant_vector = "Transformation not invertible"
            
            return {
                'original_vector': vector,
                'covariant_vector': covariant_vector,
                'contravariant_vector': contravariant_vector
            }
        
        # Example transformation
        vector = np.array([1, 2])
        transform = np.array([[2, 1], [1, 2]])
        result = coordinate_transform(vector, transform)
        print(result)
    
    @staticmetho
    def category_theory_functors():
        """
        Covariance and Contravariance in Category Theory
        
        Demonstrates functor behavior under different mappings
        """
        print("\nCategory Theory Functor Covariance/Contravariance:")
        
        # Simplified functor representation
        class Functor:
            def __init__(self, mapping):
                self.mapping = mapping
            
            def covariant_map(self, x):
                """
                Covariant functor: preserves morphism direction
                F(f ∘ g) = F(f) ∘ F(g)
                """
                return self.mapping(x)
            
            def contravariant_map(self, x):
                """
                Contravariant functor: reverses morphism direction
                F(f ∘ g) = F(g) ∘ F(f)
                """
                return self.mapping(x)[::-1]  # Simple reversal as example
        
        # Example mappings
        def square(x): return x ** 2
        
        covariant_func = Functor(square)
        contravariant_func = Functor(square)
        
        print("Covariant Map (3):", covariant_func.covariant_map(3))
        print("Contravariant Map (3):", contravariant_func.contravariant_map(3))
    
    @staticmethod
    def csharp_covariance_contravariance():
        """
        Explanation of Covariance and Contravariance in C#
        
        Provides a conceptual overview since full implementation requires C# code
        """
        print("\nC# Covariance and Contravariance:")
        print("""
        In C#, covariance and contravariance apply to generic interfaces and delegates:

        1. Covariance (out keyword):
        - Allows using a more derived type than originally specified
        - Example: IEnumerable<Derived> can be assigned to IEnumerable<Base>

        2. Contravariance (in keyword):
        - Allows using a less derived type than originally specified
        - Example: Action<Base> can be assigned to Action<Derived>

        Example Pseudo-Code:
        interface ICovariant<out T> { }
        interface IContravariant<in T> { }
        """)

def main_2():
    pass

def main_1():
    # Determinant Demonstration
    print("Determinant Calculation:")
    det_matrix = [[1, 2], [3, 4]]
    print(MatrixOperations.determinant(det_matrix))
    
    # Permanent Demonstration
    print("\nPermanent Calculation:")
    perm_matrix = [[1, 2], [3, 4]]
    print(MatrixOperations.permanent(perm_matrix))
    
    # Covariance and Contravariance Explorations
    CovarianceContravariance.mathematical_vectors()
    CovarianceContravariance.category_theory_functors()
    CovarianceContravariance.csharp_covariance_contravariance()

if __name__ == "__main__":
    main_1()
    main_2()
