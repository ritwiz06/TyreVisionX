"""Web-image curation utilities for TyreVisionX.

This package supports research-candidate image collection and review. It does
not assign ground-truth labels automatically.
"""

from src.web_collection.schemas import CandidateRecord, QuerySpec

__all__ = ["CandidateRecord", "QuerySpec"]
