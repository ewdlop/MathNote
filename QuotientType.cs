using System;

namespace QuotientTypes
{
    // Generic QuotientType class
    public class QuotientType<T>
    {
        private readonly T value;
        private readonly Func<T, T, bool> equivalence;

        public QuotientType(T value, Func<T, T, bool> equivalence)
        {
            this.value = value;
            this.equivalence = equivalence ?? throw new ArgumentNullException(nameof(equivalence));
        }

        public override bool Equals(object? obj)
        {
            if (obj is not QuotientType<T> other)
                return false;
            
            return equivalence(this.value, other.value);
        }

        public override int GetHashCode()
        {
            return value?.GetHashCode() ?? 0;
        }

        public T Value => value;
    }

    // Example usage with rational numbers
    public class Rational : IEquatable<Rational>
    {
        public int Numerator { get; }
        public int Denominator { get; }

        public Rational(int numerator, int denominator)
        {
            if (denominator == 0)
                throw new ArgumentException("Denominator cannot be zero.", nameof(denominator));

            // Normalize the rational number
            int gcd = ComputeGcd(Math.Abs(numerator), Math.Abs(denominator));
            Numerator = numerator / gcd;
            Denominator = denominator / gcd;

            // Ensure denominator is positive
            if (Denominator < 0)
            {
                Numerator = -Numerator;
                Denominator = -Denominator;
            }
        }

        private static int ComputeGcd(int a, int b)
        {
            while (b != 0)
            {
                var temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }

        public bool Equals(Rational? other)
        {
            if (other is null) return false;
            return Numerator * other.Denominator == other.Numerator * Denominator;
        }

        public override bool Equals(object? obj) => Equals(obj as Rational);

        public override int GetHashCode()
        {
            return HashCode.Combine(Numerator, Denominator);
        }

        public override string ToString() => $"{Numerator}/{Denominator}";
    }

    // Example usage with modular arithmetic
    public class ModularNumber : IEquatable<ModularNumber>
    {
        private readonly int value;
        private readonly int modulus;

        public ModularNumber(int value, int modulus)
        {
            if (modulus <= 0)
                throw new ArgumentException("Modulus must be positive.", nameof(modulus));

            this.modulus = modulus;
            this.value = ((value % modulus) + modulus) % modulus; // Ensure positive representation
        }

        public bool Equals(ModularNumber? other)
        {
            if (other is null) return false;
            if (modulus != other.modulus) return false;
            return value == other.value;
        }

        public override bool Equals(object? obj) => Equals(obj as ModularNumber);

        public override int GetHashCode()
        {
            return HashCode.Combine(value, modulus);
        }

        public override string ToString() => $"{value} (mod {modulus})";
    }

    // Example program showing usage
    class Program
    {
        static void Main()
        {
            // Example 1: Rational Numbers as Quotient Type
            var rational1 = new QuotientType<Rational>(
                new Rational(2, 4),
                (a, b) => a.Equals(b)
            );

            var rational2 = new QuotientType<Rational>(
                new Rational(1, 2),
                (a, b) => a.Equals(b)
            );

            Console.WriteLine($"Rational numbers 2/4 and 1/2 are equal: {rational1.Equals(rational2)}");

            // Example 2: Modular Arithmetic as Quotient Type
            var mod1 = new QuotientType<ModularNumber>(
                new ModularNumber(8, 5),
                (a, b) => a.Equals(b)
            );

            var mod2 = new QuotientType<ModularNumber>(
                new ModularNumber(3, 5),
                (a, b) => a.Equals(b)
            );

            Console.WriteLine($"8 ≡ 3 (mod 5): {mod1.Equals(mod2)}");

            // Example 3: Simple string quotient type based on length
            var stringQuotient1 = new QuotientType<string>(
                "hello",
                (a, b) => a.Length == b.Length
            );

            var stringQuotient2 = new QuotientType<string>(
                "world",
                (a, b) => a.Length == b.Length
            );

            Console.WriteLine($"'hello' and 'world' are equivalent (same length): {stringQuotient1.Equals(stringQuotient2)}");
        }
    }
}
