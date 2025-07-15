"""
Julia interface for GeneExpressionProgramming.jl
"""

import os
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"



from typing import Tuple, Optional, Any
import numpy as np
import juliapkg
from juliacall import Main as julia

# Global Julia interface
_julia = None
_julia_gep = None

def _initialize_julia() -> Tuple[bool, Optional[str]]:
    """Initialize Julia and load GeneExpressionProgramming.jl"""
    global _julia, _julia_gep

    if _julia is not None:
        return True, None
    
    try:
        # Resolve Julia dependencies (handles installation if needed)
        juliapkg.resolve()
        
        # Load GeneExpressionProgramming (juliacall auto-handles init)
        print("Loading GeneExpressionProgramming.jl...")
        julia.seval('using GeneExpressionProgramming')
        
        # Import Random for reproducibility
        julia.seval('using Random')
        
        _julia = julia
        _julia_gep = julia.GeneExpressionProgramming
        
        print("Julia initialization completed successfully")
        return True, None
        
    except ImportError as e:
        return False, f"Required packages not installed: {e}. Install with: pip install juliacall juliapkg"
    except Exception as e:
        return False, f"Failed to initialize Julia: {e}"

def check_julia_installation() -> Tuple[bool, str]:
    """Check if Julia and required packages are available"""
    success, error = _initialize_julia()
    if success:
        return True, "Julia and GeneExpressionProgramming.jl are available"
    else:
        return False, error or "Unknown error"

def install_julia_dependencies():
    """Install Julia dependencies (via juliapkg)"""
    print("Installing Julia dependencies...")
    try:
        juliapkg.resolve()
        print("All Julia dependencies installed successfully")
    except Exception as e:
        print(f"Error installing Julia dependencies: {e}")
        print("Ensure Julia is installed or allow auto-install via juliapkg (set PYTHON_JULIAPKG_OFFLINE=false if needed).")

class JuliaGepRegressor:
    """Julia GEP Regressor interface"""
    
    def __init__(self, number_features: int, **kwargs):
        """Initialize Julia GEP regressor"""
        success, error = _initialize_julia()
        if not success:
            raise RuntimeError(f"Failed to initialize Julia: {error}")
        
        self.number_features = number_features
        self.kwargs = kwargs
        self._julia_regressor = None
        self._fitted = False
        
        # Handle optional parameters (assign to Julia for proper conversion)
        if 'considered_dimensions' in kwargs:
            _julia.considered_dimensions = kwargs['considered_dimensions']  # Auto-converts dict to Julia Dict
        if 'max_permutations_lib' in kwargs:
            _julia.max_permutations_lib = kwargs['max_permutations_lib']
        if 'rounds' in kwargs:
            _julia.rounds = kwargs['rounds']
        
        kwargs_str = ""
        if self.kwargs:
            kwargs_str = "; " + ", ".join(f"{k}={k}" for k in self.kwargs)
        
        _julia.seval(f"regressor = GeneExpressionProgramming.GepRegressor({number_features}{kwargs_str})")
        self._julia_regressor = _julia.regressor
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            population_size: int = 1000, loss_function: str = "mse", 
            X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
            target_dimension: Optional[list] = None, **kwargs):
        """Fit the GEP regressor using Julia"""
        
        if self._julia_regressor is None:
            raise RuntimeError("Julia regressor not initialized")
        
        print(f"Fitting GEP regressor with {epochs} epochs and population size {population_size}...")
        
        # Assign numpy arrays (juliacall handles conversion, transpose for column-major)
        _julia.X_train = X.T
        _julia.y_train = y
        
        # Build fit arguments
        fit_args = [
            "regressor",
            str(epochs),
            str(population_size), 
            "X_train",
            "y_train"
        ]
        
        # Build keyword arguments (assign complex ones first)
        _julia.loss_fun = loss_function  # String is fine
        
        fit_kwargs = []
        
        if X_test is not None and y_test is not None:
            _julia.X_test = X_test.T
            _julia.y_test = y_test
            fit_kwargs.extend(["x_test=X_test", "y_test=y_test"])
            
        if target_dimension is not None:
            _julia.target_dim = target_dimension  # List to Array
            fit_kwargs.append("target_dimension=target_dim")
        
        # Add additional kwargs (assign if complex, else inline)
        for key, value in kwargs.items():
            if isinstance(value, (list, dict, np.ndarray)):
                setattr(_julia, key, value)
                fit_kwargs.append(f"{key}={key}")
            elif isinstance(value, str):
                fit_kwargs.append(f'{key}="{value}"')
            else:
                fit_kwargs.append(f'{key}={value}')
        
        # Construct and execute fit command
        fit_kwargs_str = ", ".join(fit_kwargs)
        fit_command = f"GeneExpressionProgramming.fit!({', '.join(fit_args)}; {fit_kwargs_str})"
        
        print(f"Executing: {fit_command}")
        _julia.seval(fit_command)
        
        self._fitted = True
        print("Fitting completed successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted regressor"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before making predictions")
        
        # Assign and predict
        _julia.X_pred = X.T
        _julia.seval("predictions = regressor(X_pred)")
        
        # Convert back
        predictions = np.array(_julia.predictions)
        return predictions
    
    @property
    def best_expression_(self) -> str:
        """Get the best expression as a string"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best expression")
        
        _julia.seval("best_expr = string(regressor.best_models_[1].compiled_function)")
        return str(_julia.best_expr)
    
    @property
    def best_fitness_(self) -> float:
        """Get the best fitness value"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best fitness")
        
        _julia.seval("best_fitness = regressor.best_models_[1].fitness")
        return float(_julia.best_fitness[0])
    
    def get_best_models(self, n: int = 1) -> list:
        """Get the n best models"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best models")
        
        models = []
        num_models = _julia.seval("length(regressor.best_models_)")
        for i in range(min(n, num_models)):
            _julia.seval(f"model_{i} = regressor.best_models_[{i+1}]")
            _julia.seval(f"expr_{i} = string(model_{i}.compiled_function)")
            _julia.seval(f"fitness_{i} = model_{i}.fitness")
            
            models.append({
                'expression': str(_julia.seval(f"expr_{i}")),
                'fitness': float(_julia.seval(f"fitness_{i}"))
            })
        
        return models

class JuliaGepTensorRegressor:
    """Julia GEP Tensor Regressor interface"""
    
    def __init__(self, number_features: int, gene_count: int = 2, head_len: int = 3, 
                 feature_names: Optional[list] = None, **kwargs):
        """Initialize Julia GEP tensor regressor"""
        success, error = _initialize_julia()
        if not success:
            raise RuntimeError(f"Failed to initialize Julia: {error}")
        
        self.number_features = number_features
        self.gene_count = gene_count
        self.head_len = head_len
        self.feature_names = feature_names or [f"x{i+1}" for i in range(number_features)]
        self._julia_regressor = None
        self._fitted = False
        
        # Create Julia tensor regressor
        print(f"Creating GepTensorRegressor with {number_features} features...")
        
        # Assign feature names (list to Array{String})
        _julia.feature_names = self.feature_names
        
        _julia.seval(f'''
        regressor = GeneExpressionProgramming.GepTensorRegressor(
            {number_features},
            gene_count={gene_count},
            head_len={head_len};
            feature_names=feature_names
        )
        ''')
        
        self._julia_regressor = _julia.regressor
    
    def fit(self, loss_function: str, epochs: int = 100, population_size: int = 1000):
        """Fit the tensor regressor with custom loss function name (must be defined in Julia)"""
        if self._julia_regressor is None:
            raise RuntimeError("Julia regressor not initialized")
        
        _julia.seval(f"GeneExpressionProgramming.fit!(regressor, {epochs}, {population_size}, {loss_function})")
        
        self._fitted = True
        print("Tensor fitting completed successfully")
    
    @property
    def best_expression_(self) -> str:
        """Get the best expression"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best expression")
        
        _julia.seval("GeneExpressionProgramming.print_karva_strings(regressor.best_models_[1])")
        return "Tensor expression (see Julia output)"

    @property
    def best_fitness_(self) -> float:
        """Get the best fitness value"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best fitness")
        _julia.seval("best_fitness = regressor.best_models_[1].fitness")
        return float(_julia.best_fitness)
