# DFS Refactoring Request

I have a DFS application that needs proper refactoring according to the guide in dfs/COMPLETE_APP_REFACTORING_GUIDE.md.

The previous refactoring attempt FAILED because it created new modular architecture but didn't actually EXTRACT and REMOVE the duplicate code from the original large files. This made the system more complicated instead of simpler.

Please:

1. Read the COMPLETE_APP_REFACTORING_GUIDE.md file carefully
2. Actually extract the specified code sections from the large files (data.py lines 2343-3202, models.py lines 1522-3073, etc.)
3. Remove the extracted code from the original files to reduce their size
4. Integrate the existing new modules (core/, features/, networks/, training/) that are already created or create them if they don't exist
5. Actually reduce file sizes as specified in the guide:
   - data.py: from 4022 lines down to ~3,500 lines
   - models.py: from 4436 lines down to ~2,000 lines

The goal is to make the codebase SIMPLER and SMALLER, not more complicated. Focus on actually removing duplicate code from the original files, not just creating new modules alongside them.

Do EVERYTHING properly this time - extract, remove, integrate, and validate that the large files are actually reduced in size.
