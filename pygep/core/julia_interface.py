"""
JL install helper for GeneExpressionProgramming.jl
"""

import os
import sys
from typing import Tuple, Optional, Any
import numpy as np

_julia = None
_julia_gep = None

def _initialize_julia() -> Tuple[bool, Optional[str]]:
    """Initialize Julia and load GeneExpressionProgramming.jl"""
    global _julia, _julia_gep
    
    if _julia is not None:
        return True, None
    
    try:
        # Import Julia with proper error handling
        from julia.api import Julia
        from julia import Main
        
        # Initialize Julia with compiled_modules=False to handle static linking
        print("Initializing Julia with compiled_modules=False...")
        jl = Julia(compiled_modules=False)
        _julia = Main
        
        # Install and load GeneExpressionProgramming.jl
        print("Loading GeneExpressionProgramming.jl...")
        
        # Add the package if not already installed
        _julia.eval('''
        using Pkg
        try
            using GeneExpressionProgramming
            println("✓ GeneExpressionProgramming.jl already available")
        catch
            println("Installing GeneExpressionProgramming.jl...")
            Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl")
            using GeneExpressionProgramming
            println("✓ GeneExpressionProgramming.jl installed and loaded")
        end
        ''')
        
        # Import Random for reproducibility
        _julia.eval('using Random')
        
        _julia_gep = _julia.GeneExpressionProgramming
        
        print("✓ Julia initialization completed successfully")
        return True, None
        
    except ImportError as e:
        return False, f"Julia package not installed: {e}. Install with: pip install julia"
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
    """Install Julia dependencies"""
    print("Installing Julia dependencies...")
    
    try:
        from julia import Main
        from julia.api import Julia
        
        # Initialize Julia
        jl = Julia(compiled_modules=False)
        julia = Main
        
        print("Installing GeneExpressionProgramming.jl...")
        julia.eval('''
        using Pkg
        Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl")
        using GeneExpressionProgramming
        @info "GeneExpressionProgramming.jl installed successfully"
        ''')
        
        print("Julia dependencies installed successfully")
        
    except ImportError:
        print("Julia package not found. Install with: pip install julia")
        print("Also install Julia from: https://julialang.org/downloads/")
    except Exception as e:
        print(f"Error installing Julia dependencies: {e}")

class JuliaGepRegressor:
    
    def __init__(self, number_features: int, **kwargs):
        """Initialize Julia GEP regressor"""
        success, error = _initialize_julia()
        if not success:
            raise RuntimeError(f"Failed to initialize Julia: {error}")
        
        self.number_features = number_features
        self.kwargs = kwargs
        self._julia_regressor = None
        self._fitted = False
        
        # Handle optional parameters like the working version
        julia_kwargs = []
        if 'considered_dimensions' in kwargs:
            # Convert Python dict to Julia dict for dimensions
            dims_dict = kwargs['considered_dimensions']
            julia_kwargs.append(f"considered_dimensions={dims_dict}")
        if 'max_permutations_lib' in kwargs:
            julia_kwargs.append(f"max_permutations_lib={kwargs['max_permutations_lib']}")
        if 'rounds' in kwargs:
            julia_kwargs.append(f"rounds={kwargs['rounds']}")
            
        kwargs_str = ", ".join(julia_kwargs)
        if kwargs_str:
            kwargs_str = "; " + kwargs_str
            
        _julia.eval(f"regressor = GeneExpressionProgramming.GepRegressor({number_features}{kwargs_str})")
        self._julia_regressor = _julia.regressor        
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            population_size: int = 1000, loss_function: str = "mse", 
            X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
            target_dimension: Optional[list] = None, multi_objective: bool = False, **kwargs):
        
        if self._julia_regressor is None:
            raise RuntimeError("Julia regressor not initialized")
        
        print(f"Fitting GEP regressor with {epochs} epochs and population size {population_size}...")
        
        _julia.X_train = X.T
        _julia.y_train = y
        
        fit_args = [
            "regressor",
            str(epochs),
            str(population_size), 
            "X_train",
            "y_train"
        ]
        
        fit_kwargs = [f'loss_fun="{loss_function}"']
        
        if X_test is not None and y_test is not None:
            _julia.X_test = X_test.T
            _julia.y_test = y_test
            fit_kwargs.extend(["x_test=X_test", "y_test=y_test"])
            
        if target_dimension is not None:
            _julia.target_dim = target_dimension
            fit_kwargs.append("target_dimension=target_dim")
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                fit_kwargs.append(f'{key}="{value}"')
            else:
                fit_kwargs.append(f'{key}={value}')
        
        fit_kwargs_str = ", ".join(fit_kwargs)
        fit_command = f"GeneExpressionProgramming.fit!({', '.join(fit_args)}; {fit_kwargs_str})"
        
        _julia.eval(fit_command)
        
        self._fitted = True
        print("Fitting done")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted regressor"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before making predictions")
        
        # Convert to Julia array and make predictions
        _julia.X_pred = X.T
        _julia.eval("predictions = regressor(X_pred)")
        
        # Convert back to numpy
        predictions = np.array(_julia.predictions)
        return predictions
    
    @property
    def best_expression_(self) -> str:
        """Get the best expression as a string"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best expression")
        
        _julia.eval("best_expr = string(regressor.best_models_[1].compiled_function)")
        return str(_julia.best_expr)
    
    @property
    def best_fitness_(self) -> float:
        """Get the best fitness value"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best fitness")
        
        _julia.eval("best_fitness = regressor.best_models_[1].fitness")
        return float(_julia.best_fitness)
    
    def get_best_models(self, n: int = 1) -> list:
        """Get the n best models"""
        if not self._fitted:
            raise RuntimeError("Regressor must be fitted before accessing best models")
        
        models = []
        for i in range(min(n, len(_julia.eval("regressor.best_models_")))):
            _julia.eval(f"model_{i} = regressor.best_models_[{i+1}]")
            _julia.eval(f"expr_{i} = string(model_{i}.compiled_function)")
            _julia.eval(f"fitness_{i} = model_{i}.fitness")
            
            models.append({
                'expression': str(_julia.eval(f"expr_{i}")),
                'fitness': float(_julia.eval(f"fitness_{i}"))
            })
        
        return models

def check_julia_installation() -> Tuple[bool, str]:
    """Check if Julia and GeneExpressionProgramming.jl are available"""
    try:
        # Check Julia
        result = subprocess.run(['julia', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "Julia not found or not working"
        
        # Check GeneExpressionProgramming.jl
        script = '''
using GeneExpressionProgramming
println("GeneExpressionProgramming.jl is available")
'''
        result = subprocess.run(['julia', '-e', script], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"GeneExpressionProgramming.jl not available: {result.stderr}"
        
        return True, "Julia and GeneExpressionProgramming.jl are available"
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Julia not found in PATH"

def install_julia_dependencies():
    """Install Julia dependencies"""
    print("Installing Julia dependencies...")
    
    script = '''
using Pkg
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl")
using GeneExpressionProgramming
println("✓ GeneExpressionProgramming.jl installed successfully")
'''
    
    try:
        result = subprocess.run(['julia', '-e', script], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("All Julia dependencies installed successfully")
        else:
            print(f"Error installing Julia dependencies: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Timeout installing Julia dependencies")
    except Exception as e:
        print(f"Error: {e}")

# Alias for compatibility
JuliaGepRegressor = DirectJuliaInterface

