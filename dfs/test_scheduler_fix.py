#!/usr/bin/env python
"""Test that the ReduceLROnPlateau fix works."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

def test_scheduler_fix():
    """Test that ReduceLROnPlateau works without verbose parameter."""
    try:
        # Create a simple model and optimizer
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # This should work without the verbose parameter
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            min_lr=1e-6
        )
        
        print("✅ ReduceLROnPlateau works without verbose parameter!")
        
        # Test that it can step
        loss = torch.tensor(1.0)
        scheduler.step(loss)
        
        print("✅ Scheduler can step with loss value!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_scheduler_fix()
    sys.exit(0 if success else 1)