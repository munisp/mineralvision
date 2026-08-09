"""
Data Quality Framework Module
=============================

Production-grade data quality with:
- Great Expectations-style validation
- Automated data profiling
- Quality metrics and scoring
- Anomaly detection
- Data quality dashboards
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExpectationType(Enum):
    """Types of data expectations."""
    COLUMN_EXISTS = "column_exists"
    COLUMN_TYPE = "column_type"
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    VALUE_IN_SET = "value_in_set"
    VALUE_RANGE = "value_range"
    REGEX_MATCH = "regex_match"
    CUSTOM = "custom"
    ROW_COUNT = "row_count"
    COLUMN_MEAN = "column_mean"
    COLUMN_STDDEV = "column_stddev"
    COLUMN_MIN = "column_min"
    COLUMN_MAX = "column_max"
    REFERENTIAL_INTEGRITY = "referential_integrity"


class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class Expectation:
    """Represents a data quality expectation."""
    expectation_type: ExpectationType
    column: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'expectation_type': self.expectation_type.value,
            'column': self.column,
            'kwargs': self.kwargs,
            'meta': self.meta
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Expectation':
        """Create from dictionary."""
        return cls(
            expectation_type=ExpectationType(d['expectation_type']),
            column=d.get('column'),
            kwargs=d.get('kwargs', {}),
            meta=d.get('meta', {})
        )


@dataclass
class ValidationResultDetail:
    """Detailed validation result."""
    expectation: Expectation
    success: bool
    result: ValidationResult
    observed_value: Any = None
    expected_value: Any = None
    exception_info: Optional[str] = None
    element_count: int = 0
    unexpected_count: int = 0
    unexpected_percent: float = 0.0
    partial_unexpected_list: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'expectation': self.expectation.to_dict(),
            'success': self.success,
            'result': self.result.value,
            'observed_value': self.observed_value,
            'expected_value': self.expected_value,
            'exception_info': self.exception_info,
            'element_count': self.element_count,
            'unexpected_count': self.unexpected_count,
            'unexpected_percent': self.unexpected_percent,
            'partial_unexpected_list': self.partial_unexpected_list[:10]
        }


@dataclass
class ValidationSuite:
    """Collection of expectations for validation."""
    name: str
    expectations: List[Expectation] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def add_expectation(self, expectation: Expectation):
        """Add an expectation to the suite."""
        self.expectations.append(expectation)
    
    def expect_column_to_exist(self, column: str, **kwargs):
        """Expect a column to exist."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.COLUMN_EXISTS,
            column=column,
            kwargs=kwargs
        ))
        return self
    
    def expect_column_values_to_not_be_null(self, column: str, **kwargs):
        """Expect column values to not be null."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.NOT_NULL,
            column=column,
            kwargs=kwargs
        ))
        return self
    
    def expect_column_values_to_be_unique(self, column: str, **kwargs):
        """Expect column values to be unique."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.UNIQUE,
            column=column,
            kwargs=kwargs
        ))
        return self
    
    def expect_column_values_to_be_in_set(self, column: str, value_set: List[Any], **kwargs):
        """Expect column values to be in a set."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.VALUE_IN_SET,
            column=column,
            kwargs={'value_set': value_set, **kwargs}
        ))
        return self
    
    def expect_column_values_to_be_between(self, column: str, min_value: float = None,
                                          max_value: float = None, **kwargs):
        """Expect column values to be between min and max."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.VALUE_RANGE,
            column=column,
            kwargs={'min_value': min_value, 'max_value': max_value, **kwargs}
        ))
        return self
    
    def expect_column_values_to_match_regex(self, column: str, regex: str, **kwargs):
        """Expect column values to match regex."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.REGEX_MATCH,
            column=column,
            kwargs={'regex': regex, **kwargs}
        ))
        return self
    
    def expect_table_row_count_to_be_between(self, min_value: int = None,
                                            max_value: int = None, **kwargs):
        """Expect table row count to be between min and max."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.ROW_COUNT,
            kwargs={'min_value': min_value, 'max_value': max_value, **kwargs}
        ))
        return self
    
    def expect_column_mean_to_be_between(self, column: str, min_value: float = None,
                                        max_value: float = None, **kwargs):
        """Expect column mean to be between min and max."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.COLUMN_MEAN,
            column=column,
            kwargs={'min_value': min_value, 'max_value': max_value, **kwargs}
        ))
        return self
    
    def expect_column_stdev_to_be_between(self, column: str, min_value: float = None,
                                         max_value: float = None, **kwargs):
        """Expect column standard deviation to be between min and max."""
        self.add_expectation(Expectation(
            expectation_type=ExpectationType.COLUMN_STDDEV,
            column=column,
            kwargs={'min_value': min_value, 'max_value': max_value, **kwargs}
        ))
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'expectations': [e.to_dict() for e in self.expectations],
            'meta': self.meta
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ValidationSuite':
        """Create from dictionary."""
        suite = cls(name=d['name'], meta=d.get('meta', {}))
        for exp_dict in d.get('expectations', []):
            suite.add_expectation(Expectation.from_dict(exp_dict))
        return suite
    
    def save(self, path: str):
        """Save suite to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ValidationSuite':
        """Load suite from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class DataValidator:
    """
    Data validation engine.
    """
    
    def __init__(self):
        self._custom_validators: Dict[str, Callable] = {}
    
    def register_custom_validator(self, name: str, validator_fn: Callable):
        """Register a custom validator function."""
        self._custom_validators[name] = validator_fn
    
    def validate(self, df: pd.DataFrame, suite: ValidationSuite) -> Dict[str, Any]:
        """
        Validate a DataFrame against a validation suite.
        
        Args:
            df: DataFrame to validate
            suite: Validation suite with expectations
            
        Returns:
            Validation results dictionary
        """
        results = []
        success_count = 0
        failure_count = 0
        
        for expectation in suite.expectations:
            result = self._validate_expectation(df, expectation)
            results.append(result)
            
            if result.success:
                success_count += 1
            else:
                failure_count += 1
        
        overall_success = failure_count == 0
        
        return {
            'success': overall_success,
            'suite_name': suite.name,
            'statistics': {
                'evaluated_expectations': len(results),
                'successful_expectations': success_count,
                'unsuccessful_expectations': failure_count,
                'success_percent': (success_count / len(results) * 100) if results else 0
            },
            'results': [r.to_dict() for r in results],
            'meta': {
                'validation_time': datetime.utcnow().isoformat(),
                'data_asset_name': suite.meta.get('data_asset_name', 'unknown'),
                'row_count': len(df),
                'column_count': len(df.columns)
            }
        }
    
    def _validate_expectation(self, df: pd.DataFrame,
                             expectation: Expectation) -> ValidationResultDetail:
        """Validate a single expectation."""
        try:
            if expectation.expectation_type == ExpectationType.COLUMN_EXISTS:
                return self._validate_column_exists(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.NOT_NULL:
                return self._validate_not_null(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.UNIQUE:
                return self._validate_unique(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.VALUE_IN_SET:
                return self._validate_value_in_set(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.VALUE_RANGE:
                return self._validate_value_range(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.REGEX_MATCH:
                return self._validate_regex_match(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.ROW_COUNT:
                return self._validate_row_count(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.COLUMN_MEAN:
                return self._validate_column_mean(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.COLUMN_STDDEV:
                return self._validate_column_stddev(df, expectation)
            
            elif expectation.expectation_type == ExpectationType.CUSTOM:
                return self._validate_custom(df, expectation)
            
            else:
                return ValidationResultDetail(
                    expectation=expectation,
                    success=False,
                    result=ValidationResult.SKIPPED,
                    exception_info=f"Unknown expectation type: {expectation.expectation_type}"
                )
                
        except Exception as e:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=str(e)
            )
    
    def _validate_column_exists(self, df: pd.DataFrame,
                               expectation: Expectation) -> ValidationResultDetail:
        """Validate column exists."""
        column = expectation.column
        exists = column in df.columns
        
        return ValidationResultDetail(
            expectation=expectation,
            success=exists,
            result=ValidationResult.PASSED if exists else ValidationResult.FAILED,
            observed_value=list(df.columns),
            expected_value=column
        )
    
    def _validate_not_null(self, df: pd.DataFrame,
                          expectation: Expectation) -> ValidationResultDetail:
        """Validate column values are not null."""
        column = expectation.column
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        null_count = df[column].isnull().sum()
        total_count = len(df)
        null_percent = (null_count / total_count * 100) if total_count > 0 else 0
        
        # Check mostly threshold
        mostly = expectation.kwargs.get('mostly', 1.0)
        success = (1 - null_count / total_count) >= mostly if total_count > 0 else True
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=null_percent,
            expected_value=f"<= {(1 - mostly) * 100}% null",
            element_count=total_count,
            unexpected_count=null_count,
            unexpected_percent=null_percent
        )
    
    def _validate_unique(self, df: pd.DataFrame,
                        expectation: Expectation) -> ValidationResultDetail:
        """Validate column values are unique."""
        column = expectation.column
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        total_count = len(df)
        unique_count = df[column].nunique()
        duplicate_count = total_count - unique_count
        
        mostly = expectation.kwargs.get('mostly', 1.0)
        success = (unique_count / total_count) >= mostly if total_count > 0 else True
        
        # Get duplicate values
        duplicates = df[df[column].duplicated(keep=False)][column].unique().tolist()
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=unique_count,
            expected_value=total_count,
            element_count=total_count,
            unexpected_count=duplicate_count,
            unexpected_percent=(duplicate_count / total_count * 100) if total_count > 0 else 0,
            partial_unexpected_list=duplicates[:10]
        )
    
    def _validate_value_in_set(self, df: pd.DataFrame,
                              expectation: Expectation) -> ValidationResultDetail:
        """Validate column values are in a set."""
        column = expectation.column
        value_set = set(expectation.kwargs.get('value_set', []))
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        total_count = len(df)
        in_set_mask = df[column].isin(value_set)
        unexpected_count = (~in_set_mask).sum()
        
        mostly = expectation.kwargs.get('mostly', 1.0)
        success = (in_set_mask.sum() / total_count) >= mostly if total_count > 0 else True
        
        unexpected_values = df[~in_set_mask][column].unique().tolist()
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=df[column].unique().tolist()[:20],
            expected_value=list(value_set),
            element_count=total_count,
            unexpected_count=unexpected_count,
            unexpected_percent=(unexpected_count / total_count * 100) if total_count > 0 else 0,
            partial_unexpected_list=unexpected_values[:10]
        )
    
    def _validate_value_range(self, df: pd.DataFrame,
                             expectation: Expectation) -> ValidationResultDetail:
        """Validate column values are in range."""
        column = expectation.column
        min_value = expectation.kwargs.get('min_value')
        max_value = expectation.kwargs.get('max_value')
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        total_count = len(df)
        
        # Build mask for out-of-range values
        mask = pd.Series([True] * total_count, index=df.index)
        
        if min_value is not None:
            mask &= df[column] >= min_value
        if max_value is not None:
            mask &= df[column] <= max_value
        
        unexpected_count = (~mask).sum()
        
        mostly = expectation.kwargs.get('mostly', 1.0)
        success = (mask.sum() / total_count) >= mostly if total_count > 0 else True
        
        unexpected_values = df[~mask][column].tolist()
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value={'min': df[column].min(), 'max': df[column].max()},
            expected_value={'min': min_value, 'max': max_value},
            element_count=total_count,
            unexpected_count=unexpected_count,
            unexpected_percent=(unexpected_count / total_count * 100) if total_count > 0 else 0,
            partial_unexpected_list=unexpected_values[:10]
        )
    
    def _validate_regex_match(self, df: pd.DataFrame,
                             expectation: Expectation) -> ValidationResultDetail:
        """Validate column values match regex."""
        import re
        
        column = expectation.column
        regex = expectation.kwargs.get('regex', '.*')
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        total_count = len(df)
        pattern = re.compile(regex)
        
        matches = df[column].astype(str).apply(lambda x: bool(pattern.match(x)))
        unexpected_count = (~matches).sum()
        
        mostly = expectation.kwargs.get('mostly', 1.0)
        success = (matches.sum() / total_count) >= mostly if total_count > 0 else True
        
        unexpected_values = df[~matches][column].tolist()
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=f"{matches.sum()}/{total_count} matched",
            expected_value=regex,
            element_count=total_count,
            unexpected_count=unexpected_count,
            unexpected_percent=(unexpected_count / total_count * 100) if total_count > 0 else 0,
            partial_unexpected_list=unexpected_values[:10]
        )
    
    def _validate_row_count(self, df: pd.DataFrame,
                           expectation: Expectation) -> ValidationResultDetail:
        """Validate row count is in range."""
        min_value = expectation.kwargs.get('min_value')
        max_value = expectation.kwargs.get('max_value')
        
        row_count = len(df)
        
        success = True
        if min_value is not None and row_count < min_value:
            success = False
        if max_value is not None and row_count > max_value:
            success = False
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=row_count,
            expected_value={'min': min_value, 'max': max_value},
            element_count=row_count
        )
    
    def _validate_column_mean(self, df: pd.DataFrame,
                             expectation: Expectation) -> ValidationResultDetail:
        """Validate column mean is in range."""
        column = expectation.column
        min_value = expectation.kwargs.get('min_value')
        max_value = expectation.kwargs.get('max_value')
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        mean_value = df[column].mean()
        
        success = True
        if min_value is not None and mean_value < min_value:
            success = False
        if max_value is not None and mean_value > max_value:
            success = False
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=mean_value,
            expected_value={'min': min_value, 'max': max_value},
            element_count=len(df)
        )
    
    def _validate_column_stddev(self, df: pd.DataFrame,
                               expectation: Expectation) -> ValidationResultDetail:
        """Validate column standard deviation is in range."""
        column = expectation.column
        min_value = expectation.kwargs.get('min_value')
        max_value = expectation.kwargs.get('max_value')
        
        if column not in df.columns:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Column '{column}' not found"
            )
        
        std_value = df[column].std()
        
        success = True
        if min_value is not None and std_value < min_value:
            success = False
        if max_value is not None and std_value > max_value:
            success = False
        
        return ValidationResultDetail(
            expectation=expectation,
            success=success,
            result=ValidationResult.PASSED if success else ValidationResult.FAILED,
            observed_value=std_value,
            expected_value={'min': min_value, 'max': max_value},
            element_count=len(df)
        )
    
    def _validate_custom(self, df: pd.DataFrame,
                        expectation: Expectation) -> ValidationResultDetail:
        """Validate using custom validator."""
        validator_name = expectation.kwargs.get('validator_name')
        
        if validator_name not in self._custom_validators:
            return ValidationResultDetail(
                expectation=expectation,
                success=False,
                result=ValidationResult.FAILED,
                exception_info=f"Custom validator '{validator_name}' not found"
            )
        
        validator_fn = self._custom_validators[validator_name]
        result = validator_fn(df, expectation.kwargs)
        
        return ValidationResultDetail(
            expectation=expectation,
            success=result.get('success', False),
            result=ValidationResult.PASSED if result.get('success') else ValidationResult.FAILED,
            observed_value=result.get('observed_value'),
            expected_value=result.get('expected_value'),
            element_count=len(df)
        )


class DataProfiler:
    """
    Automated data profiling.
    """
    
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Profile dictionary
        """
        profile = {
            'table_stats': self._profile_table(df),
            'column_profiles': {},
            'correlations': self._compute_correlations(df),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        for column in df.columns:
            profile['column_profiles'][column] = self._profile_column(df, column)
        
        return profile
    
    def _profile_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile table-level statistics."""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_bytes': df.memory_usage(deep=True).sum(),
            'duplicate_row_count': df.duplicated().sum(),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    def _profile_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Profile a single column."""
        series = df[column]
        dtype = str(series.dtype)
        
        profile = {
            'dtype': dtype,
            'count': len(series),
            'null_count': series.isnull().sum(),
            'null_percent': (series.isnull().sum() / len(series) * 100) if len(series) > 0 else 0,
            'unique_count': series.nunique(),
            'unique_percent': (series.nunique() / len(series) * 100) if len(series) > 0 else 0
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                'min': float(series.min()) if not series.isnull().all() else None,
                'max': float(series.max()) if not series.isnull().all() else None,
                'mean': float(series.mean()) if not series.isnull().all() else None,
                'median': float(series.median()) if not series.isnull().all() else None,
                'std': float(series.std()) if not series.isnull().all() else None,
                'quantiles': {
                    '25%': float(series.quantile(0.25)) if not series.isnull().all() else None,
                    '50%': float(series.quantile(0.50)) if not series.isnull().all() else None,
                    '75%': float(series.quantile(0.75)) if not series.isnull().all() else None
                },
                'zeros_count': (series == 0).sum(),
                'negative_count': (series < 0).sum()
            })
        
        # String columns
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                profile.update({
                    'min_length': int(non_null.str.len().min()),
                    'max_length': int(non_null.str.len().max()),
                    'mean_length': float(non_null.str.len().mean()),
                    'empty_count': (non_null == '').sum(),
                    'top_values': series.value_counts().head(10).to_dict()
                })
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                profile.update({
                    'min': str(non_null.min()),
                    'max': str(non_null.max()),
                    'range_days': (non_null.max() - non_null.min()).days
                })
        
        return profile
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find high correlations
        high_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr)
                        })
        
        return {
            'matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations
        }


class AnomalyDetector:
    """
    Data anomaly detection.
    """
    
    def detect_anomalies(self, df: pd.DataFrame, column: str,
                        method: str = 'zscore', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in a column.
        
        Args:
            df: DataFrame
            column: Column to analyze
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        if column not in df.columns:
            return {'error': f"Column '{column}' not found"}
        
        series = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(series):
            return {'error': f"Column '{column}' is not numeric"}
        
        if method == 'zscore':
            return self._detect_zscore(series, threshold)
        elif method == 'iqr':
            return self._detect_iqr(series, threshold)
        elif method == 'isolation_forest':
            return self._detect_isolation_forest(series)
        else:
            return {'error': f"Unknown method: {method}"}
    
    def _detect_zscore(self, series: pd.Series, threshold: float) -> Dict[str, Any]:
        """Detect anomalies using Z-score."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return {'anomaly_count': 0, 'anomaly_indices': [], 'anomaly_values': []}
        
        z_scores = (series - mean) / std
        anomaly_mask = abs(z_scores) > threshold
        
        return {
            'method': 'zscore',
            'threshold': threshold,
            'anomaly_count': int(anomaly_mask.sum()),
            'anomaly_percent': float(anomaly_mask.sum() / len(series) * 100),
            'anomaly_indices': series[anomaly_mask].index.tolist(),
            'anomaly_values': series[anomaly_mask].tolist(),
            'statistics': {
                'mean': float(mean),
                'std': float(std)
            }
        }
    
    def _detect_iqr(self, series: pd.Series, threshold: float = 1.5) -> Dict[str, Any]:
        """Detect anomalies using IQR."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        anomaly_mask = (series < lower_bound) | (series > upper_bound)
        
        return {
            'method': 'iqr',
            'threshold': threshold,
            'anomaly_count': int(anomaly_mask.sum()),
            'anomaly_percent': float(anomaly_mask.sum() / len(series) * 100),
            'anomaly_indices': series[anomaly_mask].index.tolist(),
            'anomaly_values': series[anomaly_mask].tolist(),
            'bounds': {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }
        }
    
    def _detect_isolation_forest(self, series: pd.Series) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            X = series.values.reshape(-1, 1)
            clf = IsolationForest(contamination=0.1, random_state=42)
            predictions = clf.fit_predict(X)
            
            anomaly_mask = predictions == -1
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': int(anomaly_mask.sum()),
                'anomaly_percent': float(anomaly_mask.sum() / len(series) * 100),
                'anomaly_indices': series[anomaly_mask].index.tolist(),
                'anomaly_values': series[anomaly_mask].tolist()
            }
            
        except ImportError:
            return {'error': 'sklearn not available for Isolation Forest'}


class DataQualityManager:
    """
    Complete data quality management.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or '/tmp/data_quality'
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.validator = DataValidator()
        self.profiler = DataProfiler()
        self.anomaly_detector = AnomalyDetector()
        
        self._suites: Dict[str, ValidationSuite] = {}
        self._results_history: List[Dict] = []
    
    def create_suite(self, name: str) -> ValidationSuite:
        """Create a new validation suite."""
        suite = ValidationSuite(name=name)
        self._suites[name] = suite
        return suite
    
    def get_suite(self, name: str) -> Optional[ValidationSuite]:
        """Get a validation suite by name."""
        return self._suites.get(name)
    
    def validate(self, df: pd.DataFrame, suite_name: str) -> Dict[str, Any]:
        """Validate DataFrame against a suite."""
        suite = self._suites.get(suite_name)
        if not suite:
            return {'error': f"Suite '{suite_name}' not found"}
        
        results = self.validator.validate(df, suite)
        
        # Store results
        self._results_history.append(results)
        self._save_results(suite_name, results)
        
        return results
    
    def profile(self, df: pd.DataFrame, name: str = None) -> Dict[str, Any]:
        """Profile a DataFrame."""
        profile = self.profiler.profile(df)
        
        if name:
            profile_path = os.path.join(self.storage_path, f"profile_{name}.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
        
        return profile
    
    def detect_anomalies(self, df: pd.DataFrame, columns: List[str] = None,
                        method: str = 'zscore') -> Dict[str, Any]:
        """Detect anomalies in DataFrame columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        for column in columns:
            results[column] = self.anomaly_detector.detect_anomalies(df, column, method)
        
        return results
    
    def _save_results(self, suite_name: str, results: Dict):
        """Save validation results."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(
            self.storage_path,
            f"validation_{suite_name}_{timestamp}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def get_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results."""
        if 'statistics' not in results:
            return 0.0
        
        return results['statistics'].get('success_percent', 0.0)


def create_quality_manager(storage_path: str = None) -> DataQualityManager:
    """Factory function to create data quality manager."""
    return DataQualityManager(storage_path)


def create_validation_suite(name: str) -> ValidationSuite:
    """Factory function to create validation suite."""
    return ValidationSuite(name=name)
