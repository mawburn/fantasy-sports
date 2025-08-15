"""CLI interface for the NFL DFS system."""

import typer

from .collect_data import app as collect_app
from .game_selection import app as game_selection_app
from .train_models import app as train_app

main = typer.Typer(help="NFL DFS System CLI")

# Add sub-applications
main.add_typer(collect_app, name="collect", help="Data collection commands")
main.add_typer(train_app, name="train", help="Model training commands")
main.add_typer(game_selection_app, name="game-selection", help="Contest recommendation commands")
