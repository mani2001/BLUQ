"""
Analysis module for verifying paper claims and generating comparison tables.
"""

from .verify_claims import ClaimsVerifier
from .merge_results import merge_results
from .generate_tables import generate_all_tables

__all__ = ['ClaimsVerifier', 'merge_results', 'generate_all_tables']
