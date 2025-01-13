from typing import Set, List, Dict
from itertools import product

class Group:
    """Simple group implementation for demonstrating cosets"""
    def __init__(self, elements: Set, operation_table: Dict):
        self.elements = elements
        self.operation_table = operation_table

    def operate(self, a, b):
        return self.operation_table[(a, b)]

def calculate_left_coset(element: any, subgroup: Set, group: Group) -> Set:
    """Calculate left coset for an element"""
    return {group.operate(element, h) for h in subgroup}

def calculate_right_coset(element: any, subgroup: Set, group: Group) -> Set:
    """Calculate right coset for an element"""
    return {group.operate(h, element) for h in subgroup}

# Example 1: Z4 (integers modulo 4)
Z4_elements = {0, 1, 2, 3}
Z4_operation = {((a, b), (a + b) % 4) for a, b in product(Z4_elements, repeat=2)}
Z4 = Group(Z4_elements, dict(Z4_operation))

# Subgroup H = {0, 2}
H = {0, 2}

print("Example 1: Z4 with subgroup H = {0, 2}")
print("Elements of Z4:", Z4_elements)
print("Subgroup H:", H)

print("\nLeft cosets:")
for g in Z4_elements:
    left_coset = calculate_left_coset(g, H, Z4)
    print(f"{g}H = {left_coset}")

print("\nRight cosets:")
for g in Z4_elements:
    right_coset = calculate_right_coset(g, H, Z4)
    print(f"H{g} = {right_coset}")

# Example 2: S3 (Symmetric group of degree 3)
# We'll represent permutations as tuples (a,b,c) meaning 1→a, 2→b, 3→c
S3_elements = {(1,2,3), (2,3,1), (3,1,2), (1,3,2), (2,1,3), (3,2,1)}

def compose_permutations(p1, p2):
    """Compose two permutations"""
    return tuple(p1[p2[i]-1] for i in range(3))

S3_operation = {((p1, p2), compose_permutations(p1, p2)) 
                for p1, p2 in product(S3_elements, repeat=2)}
S3 = Group(S3_elements, dict(S3_operation))

# Subgroup K = {(1,2,3), (1,3,2)}
K = {(1,2,3), (1,3,2)}

print("\nExample 2: S3 with subgroup K = {(1,2,3), (1,3,2)}")
print("Elements of S3:", S3_elements)
print("Subgroup K:", K)

print("\nLeft cosets:")
for g in S3_elements:
    left_coset = calculate_left_coset(g, K, S3)
    print(f"{g}K = {left_coset}")

print("\nRight cosets:")
for g in S3_elements:
    right_coset = calculate_right_coset(g, K, S3)
    print(f"K{g} = {right_coset}")

# Additional helpful functions for coset analysis
def is_normal_subgroup(subgroup: Set, group: Group) -> bool:
    """Check if a subgroup is normal (left cosets = right cosets)"""
    for g in group.elements:
        left = calculate_left_coset(g, subgroup, group)
        right = calculate_right_coset(g, subgroup, group)
        if left != right:
            return False
    return True

# Check normality for both examples
print("\nNormality check:")
print("Is H normal in Z4?", is_normal_subgroup(H, Z4))
print("Is K normal in S3?", is_normal_subgroup(K, S3))
