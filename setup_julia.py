#!/usr/bin/env python3
"""

Jl setup for GeneExpressionProgramming.jl

The script simply parses the commands into the JuliaRepl to install GeneExpressionProgramming.jl and it's debs. 

"""

import os
import sys
import subprocess
from pathlib import Path

def check_julia():
    """Check if Julia is installed and accessible."""
    try:
        result = subprocess.run(['julia', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Julia found: {result.stdout.strip()}")
            return True
        else:
            print("Julia not found or not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Julia not found in PATH")
        return False

def setup_julia_environment():
    """Set up the Julia environment for PyGEP."""
    print("=== PyGEP Julia Setup ===")
    
    # Check Julia
    if not check_julia():
        print("\nPlease install Julia from: https://julialang.org/downloads/")
        return False
    
    # Get Julia environment path
    julia_env_path = Path(__file__).parent / "pygep" / "julia_env"
    print(f"Julia environment path: {julia_env_path}")
    
    # Setup Julia environment
    julia_commands = [
        f'using Pkg; Pkg.activate("{julia_env_path}")',
        'Pkg.instantiate()',
        'Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl")',
        'using GeneExpressionProgramming',
        'println("JL environment setup complete")'
    ]
    
    for cmd in julia_commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(['julia', '-e', cmd], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return False
            else:
                print(f"Success: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            print("Timeout - Julia command took too long")
            return False
    
    print("\n=== Julia Setup Complete ===")
    return True

def main():
    """Main setup function."""
    success = setup_julia_environment()
    if success:
        print("\nYou can now use PyGEP!")
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

