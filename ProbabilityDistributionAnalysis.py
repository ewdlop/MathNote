import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class ProbabilityDistributionAnalysis:
    """
    Comprehensive analysis of Probability Density Functions (PDF)
    and Cumulative Distribution Functions (CDF)
    """
    
    @staticmethod
    def generate_distributions():
        """
        Generate multiple probability distributions
        
        Returns:
        dict: Various probability distributions
        """
        # Random seed for reproducibility
        np.random.seed(42)
        
        # Distributions to analyze
        distributions = {
            'Normal': stats.norm(loc=0, scale=1),
            'Exponential': stats.expon(scale=1),
            'Gamma': stats.gamma(a=2),
            'Beta': stats.beta(a=2, b=5)
        }
        
        return distributions
    
    @classmethod
    def compute_distribution_properties(cls, distribution):
        """
        Compute key properties of a probability distribution
        
        Parameters:
        distribution (scipy.stats.rv_continuous): Probability distribution
        
        Returns:
        dict: Distribution properties
        """
        # Generate sample data
        sample = distribution.rvs(size=10000)
        
        return {
            'mean': np.mean(sample),
            'variance': np.var(sample),
            'median': np.median(sample),
            'skewness': stats.skew(sample),
            'kurtosis': stats.kurtosis(sample)
        }
    
    @classmethod
    def pdf_cdf_relationship(cls, distribution, x_range=None):
        """
        Analyze relationship between PDF and CDF
        
        Parameters:
        distribution (scipy.stats.rv_continuous): Probability distribution
        x_range (array, optional): Range of x values to analyze
        
        Returns:
        dict: PDF and CDF relationship details
        """
        # Default x range if not provided
        if x_range is None:
            x_range = np.linspace(
                distribution.ppf(0.001),  # Percent point function (inverse of CDF)
                distribution.ppf(0.999),
                200
            )
        
        # Compute PDF and CDF
        pdf_values = distribution.pdf(x_range)
        cdf_values = distribution.cdf(x_range)
        
        # Numerical derivative of CDF to verify PDF
        numerical_pdf = np.gradient(cdf_values, x_range)
        
        return {
            'x_range': x_range,
            'pdf_values': pdf_values,
            'cdf_values': cdf_values,
            'numerical_pdf': numerical_pdf
        }
    
    @classmethod
    def inverse_transform_sampling(cls, distribution, sample_size=10000):
        """
        Demonstrate inverse transform sampling method
        
        Parameters:
        distribution (scipy.stats.rv_continuous): Probability distribution
        sample_size (int): Number of samples to generate
        
        Returns:
        array: Sampled values
        """
        # Generate uniform random numbers
        uniform_samples = np.random.uniform(0, 1, sample_size)
        
        # Apply inverse CDF (percent point function)
        samples = distribution.ppf(uniform_samples)
        
        return samples
    
    def visualize_distributions(self):
        """
        Visualize PDFs, CDFs, and their relationships
        """
        # Generate distributions
        distributions = self.generate_distributions()
        
        # Create figure with multiple subplots
        plt.figure(figsize=(16, 12))
        
        # Iterate through distributions
        subplot_positions = [
            (221, 'Normal'),
            (222, 'Exponential'),
            (223, 'Gamma'),
            (224, 'Beta')
        ]
        
        for position, dist_name in subplot_positions:
            distribution = distributions[dist_name]
            
            # Compute distribution relationship
            dist_analysis = self.pdf_cdf_relationship(distribution)
            
            # Plot setup
            plt.subplot(position)
            
            # Plot PDF
            plt.plot(
                dist_analysis['x_range'], 
                dist_analysis['pdf_values'], 
                label='PDF', 
                color='blue'
            )
            
            # Plot CDF
            plt.plot(
                dist_analysis['x_range'], 
                dist_analysis['cdf_values'], 
                label='CDF', 
                color='red', 
                linestyle='--'
            )
            
            # Plot numerical PDF derived from CDF
            plt.plot(
                dist_analysis['x_range'], 
                dist_analysis['numerical_pdf'], 
                label='Numerical PDF', 
                color='green', 
                linestyle=':'
            )
            
            plt.title(f'{dist_name} Distribution')
            plt.xlabel('X')
            plt.ylabel('Probability')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Create probability distribution analysis instance
    prob_analysis = ProbabilityDistributionAnalysis()
    
    # Generate distributions
    distributions = prob_analysis.generate_distributions()
    
    # Analyze each distribution
    print("Distribution Properties:")
    for dist_name, distribution in distributions.items():
        print(f"\n{dist_name} Distribution:")
        properties = prob_analysis.compute_distribution_properties(distribution)
        for prop, value in properties.items():
            print(f"{prop.capitalize()}: {value:.4f}")
    
    # Demonstrate PDF-CDF relationship for Normal distribution
    print("\nPDF-CDF Relationship (Normal Distribution):")
    normal_dist = distributions['Normal']
    relationship = prob_analysis.pdf_cdf_relationship(normal_dist)
    
    # Inverse Transform Sampling
    print("\nInverse Transform Sampling:")
    samples = prob_analysis.inverse_transform_sampling(normal_dist)
    print(f"Sample Mean: {np.mean(samples):.4f}")
    print(f"Sample Variance: {np.var(samples):.4f}")
    
    # Visualize distributions
    prob_analysis.visualize_distributions()

if __name__ == "__main__":
    main()
