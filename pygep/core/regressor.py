"""
Interface for GEP Regressor
"""

import numpy as np
from typing import Optional, Dict, Any, List
from .julia_interface import JuliaGepRegressor, JuliaGepTensorRegressor

class GepRegressor:
    """
    Gene Expression Programming Regressor for symbolic regression.
    
    Python interface to the Julia GeneExpressionProgramming.jl package.
    """
    
    def __init__(self, number_features: int, **kwargs):
        """
        Initialize GEP Regressor.
        
        Parameters:
        -----------
        number_features : int
            Number of input features
        **kwargs : dict
            Additional arguments passed to Julia GepRegressor:
            - considered_dimensions : dict, optional
                Physical dimensions for features (for physics-aware regression)
            - max_permutations_lib : int, optional
                Maximum permutations per tree (default: 10000)
            - rounds : int, optional
                Tree height rounds (default: 7)
        """
        self.number_features = number_features
        self._julia_regressor = JuliaGepRegressor(number_features, **kwargs)
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            population_size: int = 1000, loss_function: str = "mse",
            X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
            target_dimension: Optional[List] = None, **kwargs) -> 'GepRegressor':
        """
        Fit the GEP regressor to training data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input data
        y : np.ndarray, shape (n_samples,)
            Training target values
        epochs : int, default=1000
            Number of evolution epochs
        population_size : int, default=1000
            Size of the population
        loss_function : str, default="mse"
            Loss function to use ("mse", "mae", etc.)
        X_test : np.ndarray, optional
            Test input data for validation
        y_test : np.ndarray, optional
            Test target values for validation
        target_dimension : list, optional
            Target physical dimension for physics-aware regression
        **kwargs : dict
            Additional arguments passed to Julia fit! function
            
        Returns:
        --------
        self : GepRegressor
            Fitted regressor
        """
        # Validate inputs
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[1] != self.number_features:
            raise ValueError(f"X must have {self.number_features} features, got {X.shape[1]}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples")
        
        # Call Julia fit function
        self._julia_regressor.fit(
            X, y, epochs=epochs, population_size=population_size,
            loss_function=loss_function, X_test=X_test, y_test=y_test,
            target_dimension=target_dimension, **kwargs
        )
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted regressor.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values
        """
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.shape[1] != self.number_features:
            raise ValueError(f"X must have {self.number_features} features, got {X.shape[1]}")
        
        return self._julia_regressor.predict(X)
    
    @property
    def best_expression_(self) -> str:
        """Get the best symbolic expression found"""
        return self._julia_regressor.best_expression_
    
    @property
    def best_fitness_(self) -> float:
        """Get the fitness of the best expression"""
        return self._julia_regressor.best_fitness_
    
    def get_best_models(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get the n best models found during evolution.
        
        Parameters:
        -----------
        n : int, default=1
            Number of best models to return
            
        Returns:
        --------
        models : list of dict
            List of dictionaries containing 'expression' and 'fitness' keys
        """
        return self._julia_regressor.get_best_models(n)
    
    def get_fitness_history(self) -> Dict[str, List[float]]:
        """
        Get the fitness history during evolution.
        
        Returns:
        --------
        history : dict
            Dictionary with 'train_loss' key containing fitness values over epochs
        """
        # This would require additional Julia code to track fitness history
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing fitness history")
        
        # Placeholder
        print("Not implemented by now")
        return []

class GepTensorRegressor:
    """
    Gene Expression Programming Tensor Regressor for vector/matrix symbolic regression.
    
    This is a Python interface to the Julia GeneExpressionProgramming.jl tensor regressor.
    """
    
    def __init__(self, number_features: int, gene_count: int = 2, head_len: int = 3,
                 feature_names: Optional[List[str]] = None, **kwargs):
        """
        Initialize GEP Tensor Regressor.
        
        Parameters:
        -----------
        number_features : int
            Number of input features
        gene_count : int, default=2
            Number of genes (2 works reliably)
        head_len : int, default=3
            Head length for gene expression
        feature_names : list of str, optional
            Names for the features (e.g., ["x1", "x2", "U1", "U2", "U3"])
        """
        self.number_features = number_features
        self.gene_count = gene_count
        self.head_len = head_len
        self.feature_names = feature_names or [f"x{i+1}" for i in range(number_features)]
        
        self._julia_regressor = JuliaGepTensorRegressor(
            number_features, gene_count=gene_count, head_len=head_len,
            feature_names=self.feature_names, **kwargs
        )
        self._fitted = False
    
    def fit(self, loss_function, epochs: int = 100, population_size: int = 1000) -> 'GepTensorRegressor':
        """
        Fit the tensor regressor with a custom loss function.
        
        Parameters:
        -----------
        loss_function : callable
            Custom loss function (must be defined in Julia)
        epochs : int, default=100
            Number of evolution epochs
        population_size : int, default=1000
            Size of the population
            
        Returns:
        --------
        self : GepTensorRegressor
            Fitted regressor
        """
        self._julia_regressor.fit(loss_function, epochs=epochs, population_size=population_size)
        self._fitted = True
        return self
    
    @property
    def best_expression_(self) -> str:
        """Get the best tensor expression found"""
        return self._julia_regressor.best_expression_
    
    @property
    def best_fitness_(self) -> float:
        """Get the fitness of the best expression"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best fitness")
        return Inf
    
    def get_best_models(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get the n best tensor models"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best models")
        return [{"expression": self.best_expression_, "fitness": self.best_fitness_}]

