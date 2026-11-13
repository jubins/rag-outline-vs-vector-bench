"""Entry point for the orchestrator: uvicorn bench.main:app"""

from bench.api import create_app

app = create_app()
