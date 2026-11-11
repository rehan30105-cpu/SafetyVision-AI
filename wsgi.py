"""WSGI entry point for a production server configured outside this project."""

from app import app

__all__ = ["app"]