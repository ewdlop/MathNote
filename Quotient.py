from typing import Set, List, Dict, TypeVar, Generic,Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 1. Quotient Set Example
def quotient_set_example():
    """Example of quotient set using modulo arithmetic"""
    # Original set Z (represented as range of numbers)
    Z = range(-10, 11)
    
    # Equivalence relation: a ~ b if a ≡ b (mod 3)
    def are_equivalent(a: int, b: int) -> bool:
        return (a - b) % 3 == 0
    
    # Find equivalence classes
    classes = {}
    for x in Z:
        rep = x % 3  # Representative of the class
        if rep not in classes:
            classes[rep] = set()
        classes[rep].add(x)
    
    print("Quotient Set Z/3Z:")
    for rep, elements in classes.items():
        print(f"[{rep}] = {elements}")

# 2. Quotient Type Example
class QuotientType(Generic[T]):
    """Generic implementation of quotient types"""
    def __init__(self, value: T, equivalence: callable):
        self.value = value
        self.equivalence = equivalence
    
    def __eq__(self, other):
        if not isinstance(other, QuotientType):
            return False
        return self.equivalence(self.value, other.value)

# Example with rational numbers as quotient of integers
@dataclass
class Rational:
    num: int
    den: int
    
    def __post_init__(self):
        if self.den == 0:
            raise ValueError("Denominator cannot be zero")
        # Normalize
        gcd = self.compute_gcd(abs(self.num), abs(self.den))
        self.num //= gcd
        self.den //= gcd
        if self.den < 0:
            self.num = -self.num
            self.den = -self.den
    
    @staticmethod
    def compute_gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a
    
    def __eq__(self, other):
        return self.num * other.den == other.num * self.den

# 3. Quotient Category Example
class Category(ABC):
    """Abstract base class for categories"""
    @abstractmethod
    def objects(self) -> Set:
        pass
    
    @abstractmethod
    def morphisms(self, a: any, b: any) -> Set:
        pass
    
    @abstractmethod
    def compose(self, f: callable, g: callable) -> callable:
        pass

class Category(ABC):
    """Abstract base class for categories"""
    @abstractmethod
    def objects(self) -> Set:
        pass
    
    @abstractmethod
    def morphisms(self, a: Any, b: Any) -> Set:
        pass
    
    @abstractmethod
    def compose(self, f: Callable, g: Callable) -> Callable:
        pass

    @abstractmethod
    def identity(self, a: Any) -> Callable:
        pass

class QuotientCategory(Category):
    def __init__(self, base_category: Category, morphism_equivalence: Callable):
        """
        Initialize quotient category with base category and morphism equivalence relation
        
        Args:
            base_category: The original category
            morphism_equivalence: Function determining when two morphisms are equivalent
        """
        self.base = base_category
        self.morphism_equivalence = morphism_equivalence
        # Cache for morphism classes
        self._morphism_class_cache = {}

    def _get_morphism_classes(self, a: Set, b: Set) -> List[Set]:
        """
        Get equivalence classes of morphisms between two object classes
        
        Args:
            a: Source object equivalence class
            b: Target object equivalence class
        
        Returns:
            List of sets of equivalent morphisms
        """
        # Get all possible morphisms between any objects in the classes
        all_morphisms = set()
        for src in a:
            for tgt in b:
                all_morphisms.update(self.base.morphisms(src, tgt))
        
        # Group morphisms into equivalence classes
        classes = []
        remaining = set(all_morphisms)
        
        while remaining:
            f = remaining.pop()
            cls = {f}
            
            # Find all equivalent morphisms
            for g in list(remaining):
                if self.morphism_equivalence(f, g):
                    cls.add(g)
                    remaining.remove(g)
            
            classes.append(cls)
        
        return classes

    def _lift_morphism(self, morphism_class: Set) -> Callable:
        """
        Create a representative morphism for an equivalence class
        
        Args:
            morphism_class: Set of equivalent morphisms
        
        Returns:
            A canonical representative morphism
        """
        # Choose a representative from the class
        representative = next(iter(morphism_class))
        
        # Create a lifted morphism that works on equivalence classes
        def lifted_morphism(x: Set) -> Set:
            # Apply representative to each element of the input class
            return {representative(elem) for elem in x}
        
        return lifted_morphism

    def morphisms(self, a: Set, b: Set) -> Set:
        """
        Get all morphisms between two object classes in the quotient category
        
        Args:
            a: Source object equivalence class
            b: Target object equivalence class
        
        Returns:
            Set of morphisms between the classes
        """
        # Get morphism classes and lift each one
        classes = self._get_morphism_classes(a, b)
        return {self._lift_morphism(cls) for cls in classes}

# Example usage with a simple category
class SimpleCategory(Category):
    """Example category with integers as objects and functions as morphisms"""
    def objects(self) -> Set:
        return {0, 1, 2}  # Simple category with three objects
    
    def morphisms(self, a: int, b: int) -> Set:
        """Simple morphisms that add constants"""
        return {lambda x, k=k: (x + k) % 3 for k in range(3)}
    
    def compose(self, f: Callable, g: Callable) -> Callable:
        return lambda x: g(f(x))
    
    def identity(self, a: int) -> Callable:
        return lambda x: x

def demonstrate_quotient_morphisms():
    # Create base category
    base = SimpleCategory()
    
    # Define morphism equivalence (consider morphisms equivalent if they give same result on 0)
    def morphism_equivalence(f: Callable, g: Callable) -> bool:
        return f(0) == g(0)
    
    # Create quotient category
    quotient = QuotientCategory(base, morphism_equivalence)
    
    # Create some object classes
    class1 = frozenset({0})
    class2 = frozenset({1, 2})
    
    # Get morphisms between classes
    morphisms = quotient.morphisms(class1, class2)
    
    print("Morphisms in quotient category:")
    for i, morphism in enumerate(morphisms, 1):
        result = morphism(class1)
        print(f"Morphism {i}: {class1} → {result}")

def demonstrate_quotients():
    # 1. Quotient Set
    print("=== Quotient Set Example ===")
    quotient_set_example()
    
    # 2. Quotient Type
    print("\n=== Quotient Type Example ===")
    r1 = Rational(2, 4)   # 1/2
    r2 = Rational(3, 6)   # 1/2
    r3 = Rational(2, 3)   # 2/3
    print(f"2/4 == 3/6: {r1 == r2}")  # True
    print(f"2/4 == 2/3: {r1 == r3}")  # False
    
    # 3. Quotient Category (simplified example)
    print("\n=== Quotient Category Example ===")
    print("Consider category with Z/2Z objects and morphisms")
    print("Two morphisms are equivalent if they have same parity")

if __name__ == "__main__":
    demonstrate_quotients()
