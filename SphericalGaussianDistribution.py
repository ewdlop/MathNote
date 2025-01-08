import numpy as np
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt

class SphericalGaussianDistribution:
    """
    Spherical Gaussian (Isotropic Multivariate Gaussian) Distribution
    
    Key Characteristics:
    - Uniform covariance across all dimensions
    - Symmetric distribution on a high-dimensional sphere
    - Useful in machine learning, generative models, and dimensionality reduction
    """
    
    def __init__(self, mean, dimension, variance=1.0):
        """
        Initialize Spherical Gaussian Distribution
        
        Parameters:
        mean (array-like): Center of the distribution
        dimension (int): Number of dimensions
        variance (float): Variance along all dimensions
        """
        self.mean = np.array(mean)
        self.dimension = dimension
        self.variance = variance
        
        # Validate inputs
        if len(self.mean) != dimension:
            raise ValueError("Mean vector must match specified dimension")
    
    def pdf(self, x):
        """
        Probability Density Function
        
        Parameters:
        x (array-like): Point to evaluate probability density
        
        Returns:
        float: Probability density at x
        """
        # Standardize input
        x = np.array(x)
        
        # Compute Mahalanobis distance
        diff = x - self.mean
        squared_norm = np.sum(diff**2)
        
        # Normalization constant
        normalization = (2 * np.pi * self.variance) ** (-self.dimension / 2)
        
        # Exponential term
        exponential = np.exp(-squared_norm / (2 * self.variance))
        
        return normalization * exponential
    
    def sample(self, num_samples):
        """
        Generate samples from Spherical Gaussian Distribution
        
        Parameters:
        num_samples (int): Number of samples to generate
        
        Returns:
        numpy.ndarray: Generated samples
        """
        # Generate samples using numpy's multivariate normal
        cov = np.eye(self.dimension) * self.variance
        return np.random.multivariate_normal(
            self.mean, 
            cov, 
            num_samples
        )
    
    def visualize_distribution(self, num_samples=1000):
        """
        Visualize the distribution (for 2D and 3D cases)
        
        Parameters:
        num_samples (int): Number of samples to generate
        """
        if self.dimension not in [2, 3]:
            print("Visualization supported only for 2D and 3D distributions")
            return
        
        # Generate samples
        samples = self.sample(num_samples)
        
        # Plot
        plt.figure(figsize=(10, 6))
        if self.dimension == 2:
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
            plt.title("2D Spherical Gaussian Distribution")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
        else:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5)
            ax.set_title("3D Spherical Gaussian Distribution")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        
        plt.show()


class MultinomialDistribution:
    """
    Multinomial Distribution
    
    Key Characteristics:
    - Generalization of binomial distribution to multiple categories
    - Used for experiments with fixed number of trials and multiple possible outcomes
    - Probability mass function for categorical data
    """
    
    def __init__(self, n_trials, probabilities):
        """
        Initialize Multinomial Distribution
        
        Parameters:
        n_trials (int): Number of independent trials
        probabilities (array-like): Probability of each category
        """
        self.n_trials = n_trials
        self.probabilities = np.array(probabilities)
        
        # Validate probabilities sum to 1
        if not np.isclose(np.sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1")
    
    def pmf(self, x):
        """
        Probability Mass Function
        
        Parameters:
        x (array-like): Number of times each category occurs
        
        Returns:
        float: Probability of specific outcome
        """
        x = np.array(x)
        
        # Validate input
        if len(x) != len(self.probabilities):
            raise ValueError("Input must match number of probability categories")
        
        if np.sum(x) != self.n_trials:
            raise ValueError("Sum of category counts must equal total trials")
        
        # Compute multinomial coefficient
        coef = special.factorial(self.n_trials) / np.prod([
            special.factorial(xi) for xi in x
        ])
        
        # Compute probability
        prob = coef * np.prod(self.probabilities**x)
        
        return prob
    
    def sample(self, num_samples):
        """
        Generate samples from Multinomial Distribution
        
        Parameters:
        num_samples (int): Number of samples to generate
        
        Returns:
        numpy.ndarray: Generated samples
        """
        return np.random.multinomial(
            self.n_trials, 
            self.probabilities, 
            num_samples
        )
    
    def visualize_distribution(self, num_samples=1000):
        """
        Visualize the distribution of samples
        
        Parameters:
        num_samples (int): Number of samples to generate
        """
        samples = self.sample(num_samples)
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(self.probabilities)), 
            np.mean(samples, axis=0),
            yerr=np.std(samples, axis=0)
        )
        plt.title("Multinomial Distribution Samples")
        plt.xlabel("Categories")
        plt.ylabel("Average Occurrences")
        plt.xticks(range(len(self.probabilities)), 
                   [f"Category {i+1}" for i in range(len(self.probabilities))])
        plt.show()


def main():
    # Spherical Gaussian Distribution Example
    print("Spherical Gaussian Distribution Demonstration:")
    sg_dist = SphericalGaussianDistribution(
        mean=[0, 0], 
        dimension=2, 
        variance=1.0
    )
    print("PDF at [1, 1]:", sg_dist.pdf([1, 1]))
    
    # Multinomial Distribution Example
    print("\nMultinomial Distribution Demonstration:")
    mn_dist = MultinomialDistribution(
        n_trials=100, 
        probabilities=[0.3, 0.5, 0.2]
    )
    print("PMF for [30, 50, 20]:", mn_dist.pmf([30, 50, 20]))

if __name__ == "__main__":
    main()
