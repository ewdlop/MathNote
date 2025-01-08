import numpy as np
import sympy as sp

class VectorSpaceAnalysis:
    """
    Comprehensive analysis of covariant and contravariant spaces in vector algebra
    """
    
    @staticmethod
    def create_vector_space(basis_vectors):
        """
        Create a vector space from given basis vectors
        
        Parameters:
        basis_vectors (list): List of basis vectors
        
        Returns:
        dict: Representation of the vector space
        """
        return {
            'basis_vectors': np.array(basis_vectors),
            'dimension': len(basis_vectors)
        }
    
    @staticmethod
    def coordinate_transformation(vector_space, transformation_matrix):
        """
        Analyze vector transformation under different coordinate systems
        
        Parameters:
        vector_space (dict): Original vector space
        transformation_matrix (numpy.ndarray): Coordinate transformation matrix
        
        Returns:
        dict: Covariant and contravariant transformations
        """
        # Original basis vectors
        basis = vector_space['basis_vectors']
        
        # Covariant transformation (preserves basis relationship)
        covariant_transform = transformation_matrix @ basis.T
        
        # Contravariant transformation (uses inverse transformation)
        try:
            inv_transform = np.linalg.inv(transformation_matrix)
            contravariant_transform = inv_transform @ basis.T
        except np.linalg.LinAlgError:
            contravariant_transform = "Transformation not invertible"
        
        return {
            'original_basis': basis,
            'covariant_transformation': covariant_transform,
            'contravariant_transformation': contravariant_transform
        }
    
    @staticmethod
    def dual_space(vector_space):
        """
        Compute the dual (covector) space
        
        Parameters:
        vector_space (dict): Original vector space
        
        Returns:
        dict: Dual space representation
        """
        # Basis vectors of the original space
        basis = vector_space['basis_vectors']
        
        # Compute dual basis using gram matrix
        gram_matrix = basis @ basis.T
        
        # Compute dual basis vectors
        try:
            dual_basis = np.linalg.inv(gram_matrix) @ basis
        except np.linalg.LinAlgError:
            dual_basis = "Unable to compute dual basis"
        
        return {
            'original_space': vector_space,
            'gram_matrix': gram_matrix,
            'dual_basis': dual_basis
        }
    
    @staticmethod
    def tensor_analysis():
        """
        Symbolic analysis of tensor transformations
        Demonstrates covariant and contravariant tensor behaviors
        """
        # Symbolic variables
        x, y, z = sp.symbols('x y z')
        
        # Symbolic tensor representation
        def create_tensor():
            """
            Create a symbolic tensor with covariant and contravariant indices
            """
            # Covariant tensor (lower indices)
            covariant_tensor = sp.Matrix([
                [x, y],
                [z, x*y]
            ])
            
            # Contravariant tensor (upper indices)
            contravariant_tensor = sp.Matrix([
                [1/x, 1/y],
                [1/z, 1/(x*y)]
            ])
            
            return {
                'covariant': covariant_tensor,
                'contravariant': contravariant_tensor
            }
        
        # Tensor transformation
        def tensor_transformation(tensor):
            """
            Demonstrate tensor transformation properties
            """
            # Symbolic transformation matrix
            A = sp.Matrix([
                [sp.Symbol('a'), sp.Symbol('b')],
                [sp.Symbol('c'), sp.Symbol('d')]
            ])
            
            # Covariant tensor transformation
            covariant_transform = A * tensor['covariant'] * A.T
            
            # Contravariant tensor transformation
            contravariant_transform = A.inv() * tensor['contravariant'] * A.inv().T
            
            return {
                'covariant_transform': covariant_transform,
                'contravariant_transform': contravariant_transform
            }
        
        # Compute and return results
        tensor = create_tensor()
        transformed_tensor = tensor_transformation(tensor)
        
        return {
            'original_tensor': tensor,
            'transformed_tensor': transformed_tensor
        }

def main():
    # Create a vector space
    basis_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vector_space = VectorSpaceAnalysis.create_vector_space(basis_vectors)
    
    # Coordinate transformation
    transformation_matrix = np.array([
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 4]
    ])
    
    print("Vector Space Coordinate Transformation:")
    transformation = VectorSpaceAnalysis.coordinate_transformation(
        vector_space, 
        transformation_matrix
    )
    print("Covariant Transformation:\n", transformation['covariant_transformation'])
    
    # Dual Space Analysis
    print("\nDual Space Analysis:")
    dual_space = VectorSpaceAnalysis.dual_space(vector_space)
    print("Dual Basis:\n", dual_space['dual_basis'])
    
    # Tensor Analysis
    print("\nTensor Transformation Analysis:")
    tensor_analysis = VectorSpaceAnalysis.tensor_analysis()
    print("Tensor Transformations Computed")

if __name__ == "__main__":
    main()
