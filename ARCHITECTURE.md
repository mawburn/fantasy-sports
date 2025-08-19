# Project Architecture Plan

## Recommended File Structure for Web UI and API Expansion

```
fantasy-sports/
├── .gitignore
├── .eslintrc.js              # Root ESLint config for all JS/TS
├── .prettierrc               # Root Prettier config
├── package.json              # Root package.json for web tooling
├── CLAUDE.md
├── DK-NFLClassic-Rules.md
├── ARCHITECTURE.md           # This file
│
├── dfs/                      # Your existing core DFS system (Python)
│   ├── data.py
│   ├── models.py
│   ├── optimize.py
│   ├── run.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/
│   │   ├── nfl_dfs.db
│   │   └── DKSalaries.csv
│   ├── models/
│   │   └── *.pth
│   └── lineups/
│
├── api/                     # FastAPI backend (simplified)
│   ├── main.py              # Single file FastAPI app
│   ├── requirements.txt
│   └── .env.example
│
├── web/                     # Frontend (React/Next.js recommended)
│   ├── package.json
│   ├── next.config.js       # If using Next.js
│   ├── tailwind.config.js   # If using Tailwind
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/          # Reusable UI components
│   │   │   ├── lineup/      # Lineup display components
│   │   │   ├── players/     # Player-related components
│   │   │   └── optimization/ # Optimization UI
│   │   ├── pages/           # Next.js pages or React routes
│   │   │   ├── index.tsx    # Dashboard
│   │   │   ├── lineups/
│   │   │   ├── predictions/
│   │   │   └── contests/
│   │   ├── hooks/           # Custom React hooks
│   │   ├── services/        # API client functions
│   │   ├── types/           # TypeScript types
│   │   └── utils/
│   ├── public/
│   └── .env.example
│
└── scripts/                 # Shared scripts and utilities
    ├── deploy.sh
    ├── setup.py
    └── dev-server.sh
```

## Key Architectural Decisions

### 1. Keep `dfs/` Intact

- Your existing Python system remains unchanged
- All current functionality continues to work
- Core DFS logic stays as the engine

### 2. `api/` Backend Options

#### Option A: FastAPI Backend (Python)

- Lightweight wrapper around your DFS logic
- FastAPI for modern Python API development
- Auto-generated documentation with Pydantic

#### Option B: Fastify Backend (TypeScript)

- Node.js/TypeScript API server using Fastify
- Communicates with Python DFS system via subprocess calls
- Better integration with TypeScript frontend

### 3. `web/` as React/Next.js Frontend

- Modern web UI built with React
- TypeScript for type safety
- Server-side rendering with Next.js

### 4. Root-level Tooling

- ESLint/Prettier for all non-Python files
- Shared configuration across all JS/TS code

## API Integration Strategy

### Option A: FastAPI (Python) Integration - Single File

```python
# api/main.py - Complete FastAPI app in one file
import sys
sys.path.append('../dfs')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dfs import run

app = FastAPI(title="DFS API")

# Simple request models
class OptimizeRequest(BaseModel):
    contest_id: Optional[str] = None
    strategy: str = "balanced"
    count: int = 1

class TrainRequest(BaseModel):
    positions: Optional[List[str]] = None

# Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/predictions/{contest_id}")
async def get_predictions(contest_id: str):
    try:
        predictions = run.predict_players(contest_id)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_lineups(request: OptimizeRequest):
    try:
        run.optimize_lineups(
            contest_id=request.contest_id,
            strategy=request.strategy,
            num_lineups=request.count
        )
        return {"message": f"Generated {request.count} lineups"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(request: TrainRequest):
    try:
        run.train_models(request.positions)
        return {"message": "Training completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Option B: Fastify (TypeScript) Integration

```typescript
// api/services/dfs-service.ts
import { spawn } from "child_process";
import { promisify } from "util";
import path from "path";

export class DFSService {
  private readonly dfsPath = path.resolve(__dirname, "../../dfs");

  async getPredictions(contestId: string): Promise<any[]> {
    return this.runPythonCommand("predict", [
      "--contest-id",
      contestId,
      "--output",
      "/tmp/predictions.csv",
    ]);
  }

  async optimizeLineup(strategy: string, count: number): Promise<any[]> {
    return this.runPythonCommand("optimize", [
      "--strategy",
      strategy,
      "--count",
      count.toString(),
      "--output-dir",
      "/tmp/lineups",
    ]);
  }

  async trainModels(positions: string[]): Promise<void> {
    const positionArgs = positions.flatMap((pos) => ["--positions", pos]);
    return this.runPythonCommand("train", positionArgs);
  }

  private async runPythonCommand(
    command: string,
    args: string[]
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn("python", ["run.py", command, ...args], {
        cwd: this.dfsPath,
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      pythonProcess.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`Python process failed: ${stderr}`));
        } else {
          resolve(this.parseOutput(stdout));
        }
      });
    });
  }

  private parseOutput(output: string): any {
    // Parse Python CLI output or read generated files
    try {
      return JSON.parse(output);
    } catch {
      return { message: output.trim() };
    }
  }
}
```

## Recommended Tech Stack

### Backend (API)

#### Option A: FastAPI Stack

- **FastAPI** - Modern, fast Python API framework
- **Pydantic** - Data validation and serialization
- **Uvicorn** - ASGI server for development

#### Option B: Fastify Stack

- **Fastify** - Fast, low-overhead Node.js framework
- **TypeScript** - Type safety throughout the API
- **Zod** - Runtime type validation and parsing
- **tsx/ts-node** - TypeScript execution

### Frontend (Web)

- **Next.js** - React framework with SSR
- **TypeScript** - Type safety for JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Modern component library
- **React Query** - Data fetching and caching

### Development Tools

- **ESLint** - JavaScript/TypeScript linting
- **Prettier** - Code formatting
- **Husky** - Git hooks for pre-commit checks

## API Endpoints Structure

```
GET /api/data/contests          # List available contests
GET /api/data/players           # Get player pool for contest
GET /api/predictions/{contest}  # Get player predictions
POST /api/optimize              # Generate optimal lineups
GET /api/lineups               # List generated lineups
POST /api/train                # Trigger model training
GET /api/models/status         # Model training status
```

## Benefits of This Structure

1. **Clean Separation** - Each layer has clear responsibilities
2. **Existing Code Preserved** - Your working Python system unchanged
3. **Scalable** - Easy to add new features to any layer
4. **Modern Stack** - Industry-standard tools and frameworks
5. **Type Safety** - TypeScript frontend + Pydantic backend
6. **Development Experience** - Hot reloading, auto-docs, linting
7. **Deployment Flexibility** - Can deploy layers independently

## Implementation Steps

1. **Set up root-level tooling** (ESLint, Prettier, package.json)
2. **Create API layer** with FastAPI
3. **Build web frontend** with Next.js
4. **Integrate components** and test end-to-end
5. **Add deployment scripts**

## Environment Configuration

Each layer will have its own `.env.example`:

- `dfs/.env.example` - Database, model paths
- `api/.env.example` - API keys, CORS settings
- `web/.env.example` - API endpoints, feature flags

This architecture maintains your current working system while providing a modern, scalable path for adding web functionality.
