#!/usr/bin/env python3
"""Verify that the NFL DFS system development environment is set up correctly.

This script performs comprehensive environment validation to ensure all dependencies,
configurations, and directory structures are properly set up for development.

Why Environment Verification Matters:

1. Early Problem Detection: Catch missing dependencies before development starts
2. Onboarding Support: Help new developers get set up quickly
3. CI/CD Integration: Validate environments in automated pipelines
4. Debugging Aid: Isolate environment issues from code issues
5. Documentation: Living documentation of system requirements

Checks Performed:

- Python Version: Ensures modern Python (3.11+) for latest features
- Package Imports: Validates all required dependencies are installed
- Directory Structure: Confirms project layout is complete
- Configuration Files: Ensures essential config files are present
- PyTorch Optimization: Verifies CPU optimization settings

Usage:
    python scripts/verify_setup.py

For beginners: This is like a "system health check" that makes sure
your development environment has everything needed to run the project.
It's similar to checking that your car has fuel, oil, and working brakes
before a long trip.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version compatibility.

    Python 3.11+ is required for this project because:
    - Modern type hints (Union with | operator, Optional improvements)
    - Performance improvements (especially for CPU-intensive ML work)
    - Better error messages for debugging
    - Latest library compatibility
    - Structural pattern matching (if used in codebase)

    Version Requirements:
    - Minimum: Python 3.11 (for syntax and performance features)
    - Recommended: Python 3.11+ (latest stable features)
    - Not supported: Python 3.10 and below (missing required features)

    Returns:
        bool: True if Python version is compatible, False otherwise
    """
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    # Check for minimum required version
    if version.major == 3 and version.minor >= 11:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.11+ required")
        print("   Please upgrade Python to use modern features and performance improvements")
        return False


def check_imports():
    """Check that all required packages can be imported.

    This function validates that all essential dependencies are properly installed
    and can be imported. Missing packages would cause runtime errors.

    Package Categories:

    Core Data Science:
    - pandas: Data manipulation and analysis
    - numpy: Numerical computing foundation
    - sklearn: Machine learning algorithms and utilities

    ML Frameworks:
    - torch: PyTorch for deep learning (CPU-optimized)
    - xgboost: Gradient boosting for structured data
    - lightgbm: Fast gradient boosting alternative

    Web Framework:
    - fastapi: Modern API framework for predictions
    - pydantic: Data validation and settings

    Database:
    - sqlalchemy: Database ORM for data storage

    Domain-Specific:
    - nfl_data_py: NFL statistics data source
    - pulp: Linear programming for lineup optimization

    Operations:
    - mlflow: ML model tracking and deployment
    - loguru: Advanced logging capabilities

    Import Strategy:
    - Test each package individually to isolate failures
    - Continue checking all packages (don't stop at first failure)
    - Provide specific error messages for troubleshooting

    Returns:
        bool: True if all packages import successfully, False otherwise
    """
    # Core dependencies organized by purpose for clarity
    packages = [
        # Data Science Stack
        "torch",  # PyTorch for neural networks (CPU-optimized)
        "pandas",  # Data manipulation and analysis
        "numpy",  # Numerical computing foundation
        "sklearn",  # Machine learning utilities
        # ML Algorithms
        "xgboost",  # Gradient boosting trees
        "lightgbm",  # Fast gradient boosting
        # Web API Stack
        "fastapi",  # Modern web framework
        "pydantic",  # Data validation
        # Database
        "sqlalchemy",  # Database ORM
        # Domain-Specific
        "nfl_data_py",  # NFL statistics source
        "pulp",  # Linear programming solver
        # Operations
        "mlflow",  # ML experiment tracking
        "loguru",  # Advanced logging
    ]

    all_ok = True

    # Test each package import individually
    for package in packages:
        try:
            # Dynamic import to test package availability
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            # Provide helpful installation hints
            if package == "torch":
                print(
                    "   Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
                )
            elif package == "nfl_data_py":
                print("   Install with: pip install nfl-data-py")
            else:
                print(f"   Install with: pip install {package}")
            all_ok = False

    # Summary message
    if all_ok:
        print(f"\n✓ All {len(packages)} packages imported successfully")
    else:
        print("\n✗ Some packages failed to import - install missing dependencies")
        print("   Run: pip install -r requirements-dev.txt")

    return all_ok


def check_directories():
    """Check that project directories exist.

    Project directory structure is essential for:
    - Code organization and maintainability
    - Data storage and management
    - Model artifacts and checkpoints
    - Documentation and configuration

    Directory Structure:

    Core Code:
    - src/: Main source code modules
    - tests/: Unit and integration tests
    - scripts/: Utility and setup scripts

    Data Storage:
    - data/: All data files and storage
    - data/database/: SQLite database files
    - data/cache/: Temporary cached data
    - data/models/: Trained model artifacts
    - data/nfl_raw/: Raw NFL statistics
    - data/draftkings/: DFS salary data

    Project Management:
    - models/: Model versioning (alternative to data/models)
    - docs/architecture/: System design documentation

    Missing directories are created automatically by most operations,
    but having them present indicates proper project initialization.

    Returns:
        bool: True if all required directories exist, False otherwise
    """
    # Get project root directory (parent of scripts/)
    base_path = Path(__file__).parent.parent

    # Define expected directory structure
    directories = [
        # Core project structure
        "src",  # Main source code
        "tests",  # Test files
        "scripts",  # Utility scripts
        # Data organization
        "data",  # Data root directory
        "data/database",  # SQLite database storage
        "data/cache",  # Temporary cached files
        "data/models",  # Trained model artifacts
        "data/nfl_raw",  # Raw NFL statistics
        "data/draftkings",  # DraftKings CSV files
        # Alternative model storage
        "models",  # Model versioning directory
        # Documentation
        "docs/architecture",  # System design docs
    ]

    all_ok = True

    # Check each required directory
    for directory in directories:
        dir_path = base_path / directory
        if dir_path.exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            print(f"   Create with: mkdir -p {directory}")
            all_ok = False

    # Summary and guidance
    if all_ok:
        print(f"\n✓ All {len(directories)} directories found")
    else:
        print("\n✗ Some directories are missing")
        print("   Run: make setup  (to create all directories)")
        print("   Or manually create missing directories as shown above")

    return all_ok


def check_config_files():
    """Check that configuration files exist.

    Configuration files are essential for:
    - Environment setup and secrets management
    - Dependency management and reproducibility
    - Development workflow and automation
    - Code quality and version control

    File Categories:

    Environment Configuration:
    - .env: Local environment variables (secrets, database paths)
    - .env.example: Template for environment setup

    Dependency Management:
    - requirements.txt: Production dependencies
    - requirements-dev.txt: Development and testing dependencies
    - pyproject.toml: Project metadata and tool configuration

    Development Workflow:
    - .pre-commit-config.yaml: Removed (no longer using pre-commit)
    - .gitignore: Files to exclude from version control
    - Makefile: Automation commands for common tasks

    Documentation:
    - README.md: Project overview and setup instructions

    Missing configuration files can cause setup issues,
    development workflow problems, or security vulnerabilities.

    Returns:
        bool: True if all configuration files exist, False otherwise
    """
    # Get project root directory
    base_path = Path(__file__).parent.parent

    # Define required configuration files
    files = [
        # Environment and secrets
        ".env",  # Local environment variables
        ".env.example",  # Environment template
        # Dependencies
        "requirements.txt",  # Production dependencies
        "requirements-dev.txt",  # Development dependencies
        "pyproject.toml",  # Project configuration
        # Development tools
        # ".pre-commit-config.yaml",  # Removed (no longer using pre-commit)
        ".gitignore",  # Version control exclusions
        "Makefile",  # Task automation
        # Documentation
        "README.md",  # Project documentation
    ]

    all_ok = True

    # Check each required file
    for file in files:
        file_path = base_path / file
        if file_path.exists():
            print(f"✓ File exists: {file}")
        else:
            print(f"✗ File missing: {file}")
            # Provide specific guidance for critical files
            if file == ".env":
                print("   Create with: cp .env.example .env")
                print("   Then edit .env with your local settings")
            elif file == "requirements-dev.txt":
                print("   This file defines development dependencies")
            all_ok = False

    # Summary and guidance
    if all_ok:
        print(f"\n✓ All {len(files)} configuration files found")
    else:
        print("\n✗ Some configuration files are missing")
        print("   Run: make setup  (to create default files)")
        print("   Critical: Ensure .env file exists with proper settings")

    return all_ok


def check_torch_cpu_optimization():
    """Check PyTorch CPU optimization settings.

    This project is designed for CPU-only operation (no GPU required).
    Proper PyTorch CPU optimization is crucial for:
    - Fast model training and inference
    - Efficient use of multi-core processors
    - Reduced training time and latency
    - Cost-effective deployment (no GPU costs)

    Key Settings:
    - Thread count: How many CPU threads PyTorch uses
    - Interop threads: Parallelism between operations
    - CPU-optimized PyTorch build (MKL, OpenMP)

    Performance Impact:
    - Proper threading: 2-10x speedup on multi-core systems
    - Optimized builds: 20-50% faster operations
    - Memory efficiency: Better cache usage

    Troubleshooting:
    - If threads = 1: Threading may be disabled
    - If performance is slow: Consider torch built with MKL
    - For production: Set thread count explicitly

    Returns:
        bool: True if CPU optimization appears properly configured
    """
    import torch

    # Display current PyTorch configuration
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of CPU threads: {torch.get_num_threads()}")
    print(f"CPU cores available: {torch.get_num_interop_threads()}")

    # Check for CPU optimizations
    num_threads = torch.get_num_threads()
    interop_threads = torch.get_num_interop_threads()

    # Additional diagnostic information
    try:
        # Check if MKL (Math Kernel Library) is available for optimization
        import torch.backends.mkl

        mkl_available = torch.backends.mkl.is_available()
        print(f"MKL optimization: {'Available' if mkl_available else 'Not available'}")
    except AttributeError:
        print("MKL status: Unknown (older PyTorch version)")

    # Evaluate optimization status
    if num_threads > 1:
        print("✓ PyTorch CPU optimization enabled")
        if num_threads >= 4:
            print("   Good: Using multiple CPU threads for parallel processing")
        else:
            print("   Note: Limited thread count - consider increasing for better performance")
        return True
    else:
        print("✗ PyTorch CPU optimization may not be optimal")
        print("   Recommendation: Set torch.set_num_threads(4) or higher")
        print("   This can significantly improve training and inference speed")
        return False


def main():
    """Run all environment verification checks.

    This function orchestrates the complete environment verification process:
    1. Execute all verification checks in order
    2. Collect and summarize results
    3. Provide actionable next steps based on results
    4. Return appropriate exit code for automation

    Check Execution Order:
    - Python Version: Fundamental requirement (if this fails, others likely will too)
    - Package Imports: Core dependency validation
    - Project Directories: File system structure
    - Configuration Files: Essential config and setup files
    - PyTorch Optimization: Performance optimization settings

    Exit Codes:
    - 0: All checks passed (success)
    - 1: One or more checks failed (requires attention)

    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    # Display header with clear branding
    print("=" * 60)
    print("NFL DFS System - Environment Verification")
    print("=" * 60)
    print("Validating development environment setup...\n")

    # Define all verification checks to run
    # Order matters: fundamental checks first, optimization checks last
    checks = [
        ("Python Version", check_python_version),  # Must pass for anything else to work
        ("Package Imports", check_imports),  # Core dependencies
        ("Project Directories", check_directories),  # File system structure
        ("Configuration Files", check_config_files),  # Essential config files
        ("PyTorch CPU Optimization", check_torch_cpu_optimization),  # Performance settings
    ]

    # Execute each check and collect results
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            # Execute check function and store result
            check_result = check_func()
            results.append(check_result)
        except Exception as e:
            # Handle unexpected errors in check functions
            print(f"✗ Check failed with error: {e}")
            results.append(False)

    # Summarize results and provide guidance
    print("\n" + "=" * 60)
    passed_checks = sum(results)
    total_checks = len(results)

    if all(results):
        # All checks passed - environment is ready
        print(f"✓ All {total_checks} checks passed! Environment is ready.")
        print("\n🎉 Setup Complete - Next Steps:")
        print("1. Review docs/architecture/ for system design overview")
        print("2. Initialize database: make db-init")
        print("3. Collect initial data: make collect-data")
        print("4. Start development server: make run")
        print("5. Visit http://localhost:8000/docs for API documentation")
        return 0
    else:
        # Some checks failed - provide troubleshooting guidance
        failed_checks = total_checks - passed_checks
        print(f"✗ {failed_checks} of {total_checks} checks failed. Please fix the issues above.")
        print("\n🔧 Common Solutions:")
        print("- Install missing packages: pip install -r requirements-dev.txt")
        print("- Create missing directories: make setup")
        print("- Copy environment template: cp .env.example .env")
        print("- Update Python: Install Python 3.11 or newer")
        print("\n📞 Get Help:")
        print("- Check README.md for detailed setup instructions")
        print("- Review error messages above for specific issues")
        print("- Run individual checks to isolate problems")
        return 1


if __name__ == "__main__":
    """Script entry point.

    This allows the script to be run directly from command line:
    python scripts/verify_setup.py

    The sys.exit() call ensures the process exits with the correct
    exit code, which is important for automation and CI/CD pipelines.
    """
    sys.exit(main())
