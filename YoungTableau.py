import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class YoungTableau:
    def __init__(self, shape: List[int]):
        """
        Initialize a Young tableau with given shape
        shape: list of row lengths [row1_length, row2_length, ...]
        """
        self.shape = shape
        self.tableau = [[0] * row_len for row_len in shape]
        self.n = sum(shape)
    
    def is_valid_standard(self) -> bool:
        """Check if current tableau is a valid standard Young tableau"""
        # Check row increasing
        for row in self.tableau:
            if not all(row[i] < row[i+1] for i in range(len(row)-1) if row[i] != 0):
                return False
        
        # Check column increasing
        for j in range(max(self.shape)):
            col = [self.tableau[i][j] for i in range(len(self.tableau)) 
                  if j < len(self.tableau[i])]
            if not all(col[i] < col[i+1] for i in range(len(col)-1) if col[i] != 0):
                return False
        return True
    
    def fill_standard(self) -> bool:
        """Fill tableau with numbers 1 to n to create standard Young tableau"""
        def backtrack(pos: int) -> bool:
            if pos > self.n:
                return True
            
            for i in range(len(self.tableau)):
                for j in range(len(self.tableau[i])):
                    if self.tableau[i][j] == 0:
                        self.tableau[i][j] = pos
                        if self.is_valid_standard():
                            if backtrack(pos + 1):
                                return True
                        self.tableau[i][j] = 0
            return False
        
        return backtrack(1)
    
    def __str__(self) -> str:
        return '\n'.join([' '.join(map(str, row)) for row in self.tableau])


# Example usage:
if __name__ == "__main__":
    # Create and fill a Young tableau
    tableau = YoungTableau([3, 2, 1])
    tableau.fill_standard()
    print("Standard Young Tableau:")
    print(tableau)
    print()
