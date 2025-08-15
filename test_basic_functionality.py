#!/usr/bin/env python3
"""
Basic functionality test for the NFL DFS system components.

This script performs essential smoke tests to verify that the NFL Daily Fantasy Sports
system is properly installed and configured. Unlike comprehensive unit tests, these
are basic sanity checks that validate core functionality without requiring external
dependencies like databases or network connections.

Why Basic Functionality Testing?

1. Development Verification: Quick check that new code doesn't break basic imports
2. Environment Validation: Ensure development environment is properly configured
3. Installation Verification: Confirm all required packages are installed correctly
4. Configuration Testing: Validate that settings and configuration are accessible
5. Smoke Testing: Basic "does it work at all?" testing before deeper validation

Test Categories:

- Import Tests: Verify all core modules can be imported without errors
- Configuration Tests: Check that settings load correctly with proper defaults
- Model Structure Tests: Validate database model definitions are accessible
- Schema Tests: Ensure API response schemas are properly configured
- Feature Tests: Basic functionality of feature engineering components
- Project Structure: Verify required files and directories exist

Smoke Testing vs Unit Testing:
Smoke tests are shallow, fast tests that check basic functionality.
Unit tests are deep, comprehensive tests that verify specific behavior.
This script focuses on smoke tests - "does the system start up correctly?"

For beginners: Think of this like turning on a car and checking that
the dashboard lights up correctly before taking it for a test drive.
We're not testing performance, just that basic systems are working.

Usage:
    python test_basic_functionality.py
    
Expected outcome: All tests should pass in a properly configured environment.
"""

# Standard library imports for system operations and path handling
import sys  # System-specific parameters and functions
import os   # Operating system interface (may be used by imported modules)
from pathlib import Path  # Object-oriented filesystem paths

# Dynamic path configuration for module imports
# Add the 'src' directory to Python's module search path
# This allows importing our custom modules without installing them as packages
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """
    Test that all core system modules can be imported without errors.
    
    Import testing is the first and most critical validation step.
    If modules can't be imported, nothing else will work. This test
    verifies that:
    
    1. All required Python packages are installed
    2. Module paths are configured correctly
    3. There are no syntax errors in module files
    4. Dependencies between modules are properly resolved
    
    Modules Tested:
    - Configuration: Settings and environment management
    - Database Models: SQLAlchemy ORM definitions
    - API Schemas: Pydantic response model definitions
    - Feature Extraction: ML pipeline components
    
    Common Import Failure Causes:
    - Missing packages: pip install requirements missing
    - Python path issues: Module not found in sys.path
    - Syntax errors: Code has parsing errors
    - Circular imports: Modules importing each other
    - Missing dependencies: Required external libraries not installed
    
    Returns:
        bool: True if all imports successful, False otherwise
        
    For beginners: Import testing is like checking that all the
    tools you need for a project are available and working before
    you start building anything.
    """
    print("Testing imports...")
    
    try:
        # Test 1: Configuration System Import
        # The Settings class manages all system configuration via Pydantic
        from src.config.settings import Settings
        print("✅ Config module imported successfully")
        
        # Test 2: Database Models Import
        # Core SQLAlchemy ORM models for NFL data storage
        from src.database.models import Player, Team, Game, PlayerStats
        print("✅ Database models imported successfully")
        
        # Test 3: API Response Schema Import
        # Pydantic models that define API response structure
        from src.api.schemas import PlayerResponse, TeamResponse
        print("✅ API schemas imported successfully")
        
        # Test 4: Feature Engineering Pipeline Import
        # Core ML feature extraction components
        from src.data.processing.feature_extractor import FeatureExtractor
        print("✅ Feature extractor imported successfully")
        
        return True  # All imports successful
        
    except ImportError as e:
        # ImportError indicates missing modules, packages, or syntax errors
        print(f"❌ Import failed: {e}")
        print("   Common fixes: pip install requirements, check file paths")
        return False

def test_config():
    """
    Test that the configuration system loads correctly with proper defaults.
    
    Configuration management is critical for the NFL DFS system because:
    - Different environments need different settings (dev vs production)
    - API endpoints and database connections must be configurable
    - Performance tuning requires adjustable parameters
    - Security settings need environment-specific values
    
    Configuration Testing Goals:
    1. Settings object loads without errors
    2. Default values are reasonable and safe
    3. Required configuration keys are present
    4. Data types are correct (string, int, bool, etc.)
    5. Environment variable support is working
    
    The system uses Pydantic Settings for:
    - Type validation and coercion
    - Environment variable integration
    - Default value management
    - Configuration documentation
    
    Critical Settings Validated:
    - API host/port: Where the web server runs
    - Database URL: How to connect to data storage
    - Season data: What NFL data to load
    - Performance flags: CPU optimization settings
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    For beginners: Configuration is like the settings panel in
    a video game - it controls how the application behaves
    in different situations.
    """
    print("\nTesting configuration...")
    
    try:
        # Import the global settings instance (configured with defaults and env vars)
        from src.config.settings import settings
        
        # Test 1: API Server Configuration
        # Default localhost configuration for development
        assert settings.api_host == "127.0.0.1", f"Expected localhost, got {settings.api_host}"
        assert settings.api_port == 8000, f"Expected port 8000, got {settings.api_port}"
        print(f"   API server: {settings.api_host}:{settings.api_port}")
        
        # Test 2: Database Configuration
        # Should default to SQLite for local development
        assert settings.database_url.startswith("sqlite"), "Expected SQLite database URL"
        print(f"   Database: SQLite (local development)")
        
        # Test 3: Data Loading Configuration
        # Should load at least one season of NFL data
        assert settings.nfl_seasons_to_load >= 1, "Must load at least 1 NFL season"
        print(f"   NFL seasons to load: {settings.nfl_seasons_to_load}")
        
        print("✅ Configuration values are valid")
        return True
        
    except Exception as e:
        # Configuration errors could be missing env vars, type mismatches, etc.
        print(f"❌ Configuration test failed: {e}")
        print("   Check .env file and default settings")
        return False

def test_database_models():
    """
    Test that SQLAlchemy database models have the required structure and attributes.
    
    Database models are the foundation of data storage in the NFL DFS system.
    They define:
    - What data we can store (columns/fields)
    - How data relates to other data (relationships)
    - Data types and constraints (validation rules)
    - Database table structure (schema)
    
    Model Structure Testing:
    This test doesn't create database connections or run queries.
    Instead, it verifies that the ORM model classes have the
    expected attributes and structure.
    
    Core Models Validated:
    - Player: NFL player information (name, position, team)
    - Team: NFL team data (name, abbreviation, division)
    - Game: NFL game/schedule data (teams, date, week, season)
    - PlayerStats: Individual game performance statistics
    
    Required Attributes:
    Each model must have:
    - Primary key 'id' for unique identification
    - Foreign keys for relationships to other models
    - Core data fields specific to that entity
    
    Why This Matters:
    - Data integrity: Ensures we can store all required information
    - API functionality: API endpoints depend on these model fields
    - Feature engineering: ML pipeline needs specific data fields
    - Query capability: Database queries require these attributes
    
    Returns:
        bool: True if all models have required attributes
        
    For beginners: Database models are like forms you fill out -
    they define what information goes in each blank and how
    those forms relate to each other.
    """
    print("\nTesting database models...")
    
    try:
        # Import core SQLAlchemy models and the declarative base
        from src.database.models import Player, Team, Game, PlayerStats, Base
        
        # Test Player Model Structure
        # Essential fields for NFL player data
        assert hasattr(Player, 'id'), "Player needs primary key 'id'"
        assert hasattr(Player, 'player_id'), "Player needs NFL player_id reference"
        assert hasattr(Player, 'display_name'), "Player needs display_name for UI"
        assert hasattr(Player, 'position'), "Player needs position (QB, RB, etc.)"
        print("   Player model: Has required attributes")
        
        # Test Team Model Structure  
        # Essential fields for NFL team data
        assert hasattr(Team, 'id'), "Team needs primary key 'id'"
        assert hasattr(Team, 'team_abbr'), "Team needs abbreviation (KC, NE, etc.)"
        assert hasattr(Team, 'team_name'), "Team needs full team name"
        print("   Team model: Has required attributes")
        
        # Test Game Model Structure
        # Essential fields for NFL schedule/game data
        assert hasattr(Game, 'id'), "Game needs primary key 'id'"
        assert hasattr(Game, 'game_id'), "Game needs NFL game_id reference"
        assert hasattr(Game, 'season'), "Game needs season year"
        assert hasattr(Game, 'week'), "Game needs week number"
        print("   Game model: Has required attributes")
        
        # Test PlayerStats Model Structure
        # Essential fields for player performance data
        assert hasattr(PlayerStats, 'id'), "PlayerStats needs primary key 'id'"
        assert hasattr(PlayerStats, 'player_id'), "PlayerStats needs player reference"
        assert hasattr(PlayerStats, 'game_id'), "PlayerStats needs game reference"
        assert hasattr(PlayerStats, 'fantasy_points'), "PlayerStats needs fantasy_points calculation"
        print("   PlayerStats model: Has required attributes")
        
        print("✅ Database models have required attributes")
        return True
        
    except Exception as e:
        # Model structure errors could indicate missing fields or import issues
        print(f"❌ Database model test failed: {e}")
        print("   Check that all required model fields are defined")
        return False

def test_api_schemas():
    """
    Test that Pydantic API response schemas are properly configured.
    
    API schemas (also called response models) are crucial for:
    - Consistent API responses: Same structure every time
    - Automatic documentation: OpenAPI/Swagger docs generation
    - Data validation: Ensure responses match expected format
    - Type safety: Catch data structure errors early
    - Client integration: Frontend/mobile apps know what to expect
    
    Pydantic Schema Benefits:
    1. Automatic serialization: Convert database objects to JSON
    2. Field validation: Ensure data types and formats are correct
    3. Documentation: Self-documenting with field descriptions
    4. IDE support: Type hints for better development experience
    5. Error handling: Clear error messages for malformed data
    
    What We're Testing:
    - Schema classes can be imported successfully
    - Pydantic configuration is properly set up
    - Required schemas exist for all major data types
    
    Schema Categories:
    - Player data: PlayerResponse for player information
    - Team data: TeamResponse for team information  
    - Game data: GameResponse for schedule/results
    - Statistics: PlayerStatsResponse for performance data
    - DraftKings: ContestResponse and SalaryResponse for DFS data
    
    Returns:
        bool: True if all schemas are properly configured
        
    For beginners: API schemas are like templates that ensure
    all API responses have the same format, like how all
    restaurant receipts have the same layout.
    """
    print("\nTesting API schemas...")
    
    try:
        # Import all major Pydantic response schema classes
        from src.api.schemas import (
            PlayerResponse,      # Schema for player data API responses
            TeamResponse,        # Schema for team data API responses
            GameResponse,        # Schema for game data API responses
            PlayerStatsResponse, # Schema for player statistics API responses
            ContestResponse,     # Schema for DraftKings contest API responses
            SalaryResponse       # Schema for DraftKings salary API responses
        )
        print("   All schema classes imported successfully")
        
        # Test Pydantic Configuration
        # model_config is required for proper Pydantic v2 operation
        assert hasattr(PlayerResponse, 'model_config'), "PlayerResponse needs model_config"
        assert hasattr(TeamResponse, 'model_config'), "TeamResponse needs model_config"
        assert hasattr(GameResponse, 'model_config'), "GameResponse needs model_config"
        print("   Pydantic model configuration: Present")
        
        # Additional validation: Check that schemas are actually Pydantic models
        # This ensures they have the expected serialization and validation behavior
        from pydantic import BaseModel
        assert issubclass(PlayerResponse, BaseModel), "PlayerResponse must inherit from BaseModel"
        assert issubclass(TeamResponse, BaseModel), "TeamResponse must inherit from BaseModel"
        print("   Schema inheritance: Proper Pydantic BaseModel inheritance")
        
        print("✅ API schemas are properly configured")
        return True
        
    except Exception as e:
        # Schema errors could indicate missing imports, wrong Pydantic version, etc.
        print(f"❌ API schema test failed: {e}")
        print("   Check Pydantic version and schema definitions")
        return False

def test_feature_extractor():
    """
    Test the machine learning feature extraction pipeline structure and functionality.
    
    Feature extraction is the bridge between raw NFL data and ML predictions.
    It transforms statistical data into numerical features that machine learning
    models can understand and learn from.
    
    What Feature Extraction Does:
    1. Player Features: Recent performance, consistency, matchup analysis
    2. Team Features: Offensive/defensive rankings, home/away performance
    3. Game Features: Weather, betting lines, pace factors
    4. Rolling Statistics: Moving averages, trend analysis
    5. Fantasy Scoring: Convert raw stats to fantasy points
    
    Components Being Tested:
    - FeatureExtractor Class: Main feature engineering pipeline
    - Fantasy Points Calculation: Core scoring algorithm
    - Required Methods: All essential feature extraction functions
    
    Fantasy Points Calculation Test:
    We test the fantasy points calculation with known inputs to ensure
    the scoring algorithm works correctly. This uses standard fantasy
    football scoring rules:
    - Passing: 1 point per 25 yards, 4 points per TD, -2 per interception
    - Rushing: 1 point per 10 yards, 6 points per TD
    - Receiving: 1 point per 10 yards, 6 points per TD, 1 point per reception (PPR)
    
    Example Test Case:
    QB with 300 passing yards, 2 passing TDs, 1 interception, 50 rushing yards, 1 rushing TD:
    = (300 * 0.04) + (2 * 4) + (-1 * 2) + (50 * 0.1) + (1 * 6) = 12 + 8 - 2 + 5 + 6 = 29 points
    
    Returns:
        bool: True if feature extraction components work correctly
        
    For beginners: Feature extraction is like creating a player scouting
    report - taking raw game statistics and turning them into meaningful
    insights that help predict future performance.
    """
    print("\nTesting feature extractor...")
    
    try:
        # Import feature extraction components
        from src.data.processing.feature_extractor import FeatureExtractor, calculate_fantasy_points
        print("   Feature extraction modules imported successfully")
        
        # Test FeatureExtractor Class Structure
        # Verify all essential feature extraction methods are present
        assert hasattr(FeatureExtractor, 'extract_player_features'), "Missing extract_player_features method"
        assert hasattr(FeatureExtractor, 'extract_team_features'), "Missing extract_team_features method"
        assert hasattr(FeatureExtractor, 'extract_slate_features'), "Missing extract_slate_features method"
        print("   FeatureExtractor class: Has all required methods")
        
        # Test Fantasy Points Calculation Algorithm
        # Create realistic test case: QB performance with mixed results
        test_stats = {
            'passing_yards': 300,        # 12 points (300 * 0.04)
            'passing_tds': 2,            # 8 points (2 * 4)
            'passing_interceptions': 1,  # -2 points (1 * -2)
            'rushing_yards': 50,         # 5 points (50 * 0.1)
            'rushing_tds': 1,            # 6 points (1 * 6)
            'receiving_yards': 0,        # 0 points (QBs don't receive)
            'receiving_tds': 0,          # 0 points
            'receptions': 0,             # 0 points (PPR doesn't apply to QBs)
            'fumbles_lost': 0,           # 0 points (no fumbles)
            'two_point_conversions': 0   # 0 points (no 2-pt conversions)
        }
        
        # Calculate fantasy points using our algorithm
        points = calculate_fantasy_points(test_stats, "standard")
        
        # Calculate expected points manually for verification
        # Standard scoring: pass 1/25 yards, pass TD 4pts, INT -2pts, rush 1/10 yards, rush TD 6pts
        expected_points = (300 * 0.04) + (2 * 4) + (-1 * 2) + (50 * 0.1) + (1 * 6)
        # = 12 + 8 - 2 + 5 + 6 = 29 points
        
        # Verify calculation is accurate (within floating point tolerance)
        assert abs(points - expected_points) < 0.01, f"Fantasy points calculation error: expected {expected_points}, got {points}"
        print(f"   Fantasy points calculation: {points} points (correct)")
        
        print("✅ Feature extractor has required methods and fantasy calculation works")
        return True
        
    except Exception as e:
        # Feature extraction errors could indicate missing methods, calculation errors, etc.
        print(f"❌ Feature extractor test failed: {e}")
        print("   Check feature extraction implementation and fantasy scoring algorithm")
        return False

def test_project_structure():
    """
    Test that all required files and directories exist for proper system operation.
    
    Project structure validation ensures that:
    - All essential Python modules are present
    - Package structure follows Python conventions
    - No critical files are missing from the repository
    - Development setup is complete
    
    Why Project Structure Matters:
    1. Import Dependencies: Python needs __init__.py files to treat directories as packages
    2. Module Organization: Logical file organization makes code maintainable
    3. API Functionality: Missing router files break API endpoints
    4. Data Pipeline: Missing collection/processing files break data flows
    5. Development Workflow: Missing CLI files break command-line operations
    
    File Categories Being Validated:
    - Package Markers: __init__.py files for Python package recognition
    - Configuration: Settings and environment management files
    - Database: ORM models, connections, and initialization scripts
    - API: FastAPI application, routers, and response schemas
    - Data Processing: Collection and feature extraction modules
    - CLI Tools: Command-line interfaces for system management
    
    Python Package Structure:
    Every directory containing Python modules should have an __init__.py file
    (even if empty) to be recognized as a package. This enables:
    - Relative imports between modules
    - Package-level initialization code
    - Proper module discovery by Python
    
    Returns:
        bool: True if all required files exist, False otherwise
        
    For beginners: Project structure is like organizing a workshop -
    having tools in the right places makes everything work smoothly.
    Missing essential tools (files) means certain tasks won't work.
    """
    print("\nTesting project structure...")
    
    # Comprehensive list of required files for full system functionality
    required_files = [
        # Core package structure
        "src/__init__.py",                                  # Main package marker
        
        # Configuration system
        "src/config/__init__.py",                           # Config package marker
        "src/config/settings.py",                           # Pydantic settings management
        
        # Database layer
        "src/database/__init__.py",                         # Database package marker
        "src/database/models.py",                           # SQLAlchemy ORM models
        "src/database/connection.py",                       # Database connection management
        "src/database/init_db.py",                          # Database initialization script
        
        # API layer
        "src/api/__init__.py",                              # API package marker
        "src/api/main.py",                                  # FastAPI application entry point
        "src/api/routers/__init__.py",                      # API routers package marker
        "src/api/routers/data.py",                          # Data access API endpoints
        "src/api/schemas.py",                               # Pydantic response schemas
        
        # Data processing layer
        "src/data/__init__.py",                             # Data package marker
        "src/data/collection/__init__.py",                  # Data collection package marker
        "src/data/collection/nfl_collector.py",             # NFL data collection pipeline
        "src/data/processing/__init__.py",                  # Data processing package marker
        "src/data/processing/feature_extractor.py",         # ML feature engineering pipeline
        
        # Command-line interface
        "src/cli/__init__.py",                              # CLI package marker
        "src/cli/collect_data.py",                          # Data collection CLI commands
    ]
    
    # Check each required file for existence
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    # Report results with detailed feedback
    if missing_files:
        print(f"❌ Missing required files ({len(missing_files)} total):")
        for missing_file in missing_files:
            print(f"   - {missing_file}")
        print("   Fix: Ensure all required modules are implemented")
        return False
    
    print(f"✅ All {len(required_files)} required files exist")
    print("   Project structure: Complete")
    return True

def main():
    """
    Execute all basic functionality tests and provide comprehensive results.
    
    This main function orchestrates the complete test suite and provides
    detailed feedback about system readiness. It's designed to give
    developers quick confidence that the basic system is working.
    
    Test Execution Strategy:
    1. Run all tests regardless of individual failures (collect all issues)
    2. Provide detailed feedback for each test category
    3. Generate summary statistics for overall assessment
    4. Return appropriate exit codes for automation integration
    
    Test Categories (in execution order):
    1. Project Structure: Verify all required files exist
    2. Import Tests: Ensure all modules can be loaded
    3. Configuration: Validate settings and environment
    4. Database Models: Check ORM model structure
    5. API Schemas: Verify response model definitions
    6. Feature Extraction: Test ML pipeline components
    
    Success Criteria:
    - All tests must pass for the system to be considered ready
    - Any single failure indicates a setup or implementation issue
    - Detailed error messages help identify specific problems
    
    Exit Codes:
    - 0: All tests passed (system ready for use)
    - 1: One or more tests failed (system needs fixes)
    
    Output Format:
    - Real-time test results with ✅/❌ indicators
    - Final summary with pass/fail counts
    - Clear success/failure message
    - Actionable recommendations for failures
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
        
    For beginners: This is like running a complete diagnostic check
    on a system before using it - making sure all the basic components
    are working correctly before doing anything important.
    """
    print("=" * 60)
    print("NFL DFS System - Basic Functionality Tests")
    print("=" * 60)
    
    # Define test execution order (logical dependency order)
    # Run foundational tests first, then build up to higher-level functionality
    tests = [
        test_project_structure,   # Must have files before importing
        test_imports,            # Must import before testing functionality
        test_config,             # Test configuration after imports work
        test_database_models,    # Test database structure after imports
        test_api_schemas,        # Test API schemas after database models
        test_feature_extractor,  # Test ML components after basic structure
    ]
    
    # Execute each test with comprehensive error handling
    results = []
    for test in tests:
        try:
            print(f"\nRunning {test.__name__.replace('test_', '').replace('_', ' ').title()}...")
            result = test()  # Execute the test function
            results.append(result)
        except Exception as e:
            # Handle unexpected errors (should be rare with proper test design)
            print(f"❌ Test {test.__name__} crashed with unexpected error: {e}")
            print("   This indicates a serious implementation issue")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    # Calculate and display comprehensive test results
    passed = sum(results)
    total = len(results)
    
    # Individual test results with clear status indicators
    print("\nIndividual Test Results:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"  {i+1}. {test_name}: {status}")
    
    # Summary statistics
    print(f"\nTest Summary: {passed}/{total} tests passed ({(passed/total)*100:.0f}% success rate)")
    
    # Final assessment with actionable guidance
    if passed == total:
        print("\n🎉 All tests passed! The basic system structure is working correctly.")
        print("🚀 System is ready for:")  
        print("   - Database initialization (python -m src.cli.collect_data init-db)")
        print("   - Data collection (python -m src.cli.collect_data collect-all)")
        print("   - API server startup (uvicorn src.api.main:app --reload)")
        return 0  # Success exit code
    else:
        failed_count = total - passed
        print(f"\n⚠️  {failed_count} test{'s' if failed_count > 1 else ''} failed.")
        print("🔧 Recommended fixes:")
        print("   1. Check that all required packages are installed (uv pip install -r requirements-dev.txt)")
        print("   2. Verify Python path configuration")
        print("   3. Review error messages above for specific issues")
        print("   4. Ensure all source files are properly implemented")
        return 1  # Failure exit code

# Standard Python idiom for executable script
# This allows the file to be both imported as a module and executed directly
if __name__ == "__main__":
    # Use sys.exit() to ensure proper exit code propagation to shell
    # This enables integration with CI/CD pipelines, build scripts, and automation
    sys.exit(main())