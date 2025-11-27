"""
Input validation and sanitization utilities.

Provides additional validation beyond Pydantic schemas.
"""
import re
from typing import Optional
from fastapi import HTTPException, status


class InputValidator:
    """Input validation utilities for security."""
    
    # Regex patterns for validation
    ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9_-]+$')
    EMAIL = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    # Dangerous patterns (SQL injection, XSS, etc.)
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]
    
    @staticmethod
    def validate_alphanumeric(value: str, field_name: str = "field") -> str:
        """Validate that string contains only alphanumeric characters."""
        if not InputValidator.ALPHANUMERIC.match(value):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must contain only alphanumeric characters, hyphens, and underscores"
            )
        return value
    
    @staticmethod
    def validate_length(
        value: str, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "field"
    ) -> str:
        """Validate string length."""
        if min_length and len(value) < min_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be at least {min_length} characters"
            )
        
        if max_length and len(value) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must not exceed {max_length} characters"
            )
        
        return value
    
    @staticmethod
    def sanitize_sql(value: str) -> str:
        """
        Check for SQL injection patterns.
        
        Note: This is defense in depth - primary protection is parameterized queries.
        """
        value_upper = value.upper()
        
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected. Please check your input and try again."
                )
        
        return value
    
    @staticmethod
    def sanitize_xss(value: str) -> str:
        """Check for XSS patterns."""
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid input detected. HTML/JavaScript not allowed."
                )
        
        return value
    
    @staticmethod
    def validate_range(
        value: int | float,
        min_value: Optional[int | float] = None,
        max_value: Optional[int | float] = None,
        field_name: str = "field"
    ) -> int | float:
        """Validate numeric range."""
        if min_value is not None and value < min_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be at least {min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must not exceed {max_value}"
            )
        
        return value
    
    @staticmethod
    def validate_enum(value: str, allowed_values: list, field_name: str = "field") -> str:
        """Validate that value is in allowed list."""
        if value not in allowed_values:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be one of: {', '.join(allowed_values)}"
            )
        return value


# Convenience instance
validator = InputValidator()
