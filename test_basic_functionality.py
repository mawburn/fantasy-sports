#!/usr/bin/env python3
"""Basic functionality test for the NFL DFS system components."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test config
        from src.config.settings import Settings
        print("✅ Config module imported successfully")
        
        # Test database models
        from src.database.models import Player, Team, Game, PlayerStats
        print("✅ Database models imported successfully")
        
        # Test API schemas
        from src.api.schemas import PlayerResponse, TeamResponse
        print("✅ API schemas imported successfully")
        
        # Test feature extractor (basic structure)
        from src.data.processing.feature_extractor import FeatureExtractor
        print("✅ Feature extractor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration setup."""
    print("\nTesting configuration...")
    
    try:
        from src.config.settings import settings
        
        # Test basic config values
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8000
        assert settings.database_url.startswith("sqlite")
        assert settings.nfl_seasons_to_load >= 1
        
        print("✅ Configuration values are valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_models():
    """Test database model structure."""
    print("\nTesting database models...")
    
    try:
        from src.database.models import Player, Team, Game, PlayerStats, Base
        
        # Test that models have required attributes
        assert hasattr(Player, 'id')
        assert hasattr(Player, 'player_id')
        assert hasattr(Player, 'display_name')
        assert hasattr(Player, 'position')
        
        assert hasattr(Team, 'id')
        assert hasattr(Team, 'team_abbr')
        assert hasattr(Team, 'team_name')
        
        assert hasattr(Game, 'id')
        assert hasattr(Game, 'game_id')
        assert hasattr(Game, 'season')
        assert hasattr(Game, 'week')
        
        assert hasattr(PlayerStats, 'id')
        assert hasattr(PlayerStats, 'player_id')
        assert hasattr(PlayerStats, 'game_id')
        assert hasattr(PlayerStats, 'fantasy_points')
        
        print("✅ Database models have required attributes")
        return True
        
    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False

def test_api_schemas():
    """Test API schema structure."""
    print("\nTesting API schemas...")
    
    try:
        from src.api.schemas import (
            PlayerResponse, TeamResponse, GameResponse, 
            PlayerStatsResponse, ContestResponse, SalaryResponse
        )
        
        # Test that schemas have model_config
        assert hasattr(PlayerResponse, 'model_config')
        assert hasattr(TeamResponse, 'model_config')
        assert hasattr(GameResponse, 'model_config')
        
        print("✅ API schemas are properly configured")
        return True
        
    except Exception as e:
        print(f"❌ API schema test failed: {e}")
        return False

def test_feature_extractor():
    """Test feature extractor class structure."""
    print("\nTesting feature extractor...")
    
    try:
        from src.data.processing.feature_extractor import FeatureExtractor, calculate_fantasy_points
        
        # Test that class has required methods
        assert hasattr(FeatureExtractor, 'extract_player_features')
        assert hasattr(FeatureExtractor, 'extract_team_features')
        assert hasattr(FeatureExtractor, 'extract_slate_features')
        
        # Test fantasy points calculation
        test_stats = {
            'passing_yards': 300,
            'passing_tds': 2,
            'passing_interceptions': 1,
            'rushing_yards': 50,
            'rushing_tds': 1,
            'receiving_yards': 0,
            'receiving_tds': 0,
            'receptions': 0,
            'fumbles_lost': 0,
            'two_point_conversions': 0
        }
        
        points = calculate_fantasy_points(test_stats, "standard")
        expected_points = (300 * 0.04) + (2 * 4) - (1 * 2) + (50 * 0.1) + (1 * 6)
        assert abs(points - expected_points) < 0.01, f"Expected {expected_points}, got {points}"
        
        print("✅ Feature extractor has required methods and fantasy calculation works")
        return True
        
    except Exception as e:
        print(f"❌ Feature extractor test failed: {e}")
        return False

def test_project_structure():
    """Test that required directories and files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/__init__.py",
        "src/config/__init__.py",
        "src/config/settings.py",
        "src/database/__init__.py",
        "src/database/models.py",
        "src/database/connection.py",
        "src/database/init_db.py",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/api/routers/__init__.py",
        "src/api/routers/data.py",
        "src/api/schemas.py",
        "src/data/__init__.py",
        "src/data/collection/__init__.py",
        "src/data/collection/nfl_collector.py",
        "src/data/processing/__init__.py",
        "src/data/processing/feature_extractor.py",
        "src/cli/__init__.py",
        "src/cli/collect_data.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("NFL DFS System - Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_imports,
        test_config,
        test_database_models,
        test_api_schemas,
        test_feature_extractor,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The basic system structure is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())