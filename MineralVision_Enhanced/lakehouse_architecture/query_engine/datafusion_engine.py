"""
Apache DataFusion Query Engine for MineralVision

This module implements the Apache DataFusion query engine for the MineralVision Lakehouse architecture.
It provides functionality for high-performance SQL queries, data analysis, and
optimized processing of geospatial data.

Uses DataFusion when available, with DuckDB/pandas fallback for local processing.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    import datafusion
    from datafusion import SessionContext, col, lit
    DATAFUSION_AVAILABLE = True
except ImportError:
    DATAFUSION_AVAILABLE = False
    datafusion = None
    SessionContext = None

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

@dataclass
class DataFusionConfig:
    """Configuration settings for Apache DataFusion query engine."""
    batch_size: int = 8192
    concurrency: int = 4
    memory_limit: Optional[int] = None  # Memory limit in bytes
    parquet_pruning: bool = True
    repartition_joins: bool = True
    repartition_aggregations: bool = True
    repartition_windows: bool = True
    log_level: str = "INFO"
    
    # Geospatial-specific configurations
    enable_geospatial: bool = True
    spatial_partitioning: bool = True
    
    def __post_init__(self):
        """Initialize logging."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("DataFusionEngine")
        self.logger.info("Initialized DataFusion configuration")


class DataFusionEngine:
    """
    Main class for Apache DataFusion query engine in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Executing SQL queries with high performance
    - Creating and managing DataFrames
    - Optimizing query execution plans
    - Processing geospatial queries
    - Integrating with Delta Lake and Parquet data sources
    """
    
    def __init__(self, config: DataFusionConfig):
        """
        Initialize the DataFusion query engine.
        
        Args:
            config: Configuration settings for DataFusion query engine
        """
        self.config = config
        self.logger = logging.getLogger("DataFusionEngine")
        self._ctx = None
        self._duckdb_conn = None
        self._tables: Dict[str, str] = {}
        
        if DATAFUSION_AVAILABLE:
            try:
                self._ctx = SessionContext()
                self.logger.info("Initialized DataFusion query engine")
            except Exception as e:
                self.logger.warning(f"Could not initialize DataFusion: {e}. Using fallback.")
                self._ctx = None
        
        if self._ctx is None and DUCKDB_AVAILABLE:
            try:
                self._duckdb_conn = duckdb.connect(':memory:')
                self.logger.info("Initialized DuckDB fallback query engine")
            except Exception as e:
                self.logger.warning(f"Could not initialize DuckDB: {e}. Using pandas fallback.")
                self._duckdb_conn = None
        
        if self._ctx is None and self._duckdb_conn is None:
            self.logger.info("Using pandas-based SQL fallback")
    
    def register_table(self, name: str, path: Any, format_type: str = "parquet",
                      options: Optional[Dict[str, str]] = None) -> bool:
        """
        Register a table in the DataFusion context.

        Args:
            name: Name of the table
            path: Path to the data, or an in-memory pandas DataFrame
            format_type: Format of the data (parquet, csv, etc.)
            options: Additional options for reading the data

        Returns:
            bool: True if table was registered successfully
        """
        self.logger.info(f"Registering table '{name}' with format {format_type}")

        try:
            self._tables[name] = path

            if isinstance(path, pd.DataFrame):
                if self._duckdb_conn is not None:
                    self._duckdb_conn.register(name, path)
                elif self._ctx is not None:
                    self._ctx.register_dataframe(name, path)
                self.logger.info(f"Successfully registered DataFrame table '{name}'")
                return True

            if self._ctx is not None:
                if format_type.lower() == 'parquet':
                    self._ctx.register_parquet(name, path)
                elif format_type.lower() == 'csv':
                    self._ctx.register_csv(name, path)
                else:
                    self._ctx.register_parquet(name, path)
            elif self._duckdb_conn is not None:
                if format_type.lower() == 'parquet':
                    self._duckdb_conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
                elif format_type.lower() == 'csv':
                    self._duckdb_conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
                else:
                    self._duckdb_conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
            
            self.logger.info(f"Successfully registered table '{name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register table '{name}': {str(e)}")
            return False
    
    def execute_sql(self, query: str) -> Tuple[List[str], List[List[Any]]]:
        """
        Execute a SQL query using DataFusion or fallback engine.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple[List[str], List[List[Any]]]: Column names and rows of the result
        """
        self.logger.info(f"Executing SQL query: {query[:100]}...")
        
        try:
            if self._ctx is not None:
                result = self._ctx.sql(query)
                df = result.to_pandas()
                columns = list(df.columns)
                rows = df.values.tolist()
                self.logger.info(f"Executed query using DataFusion, returned {len(rows)} rows")
                return (columns, rows)
            elif self._duckdb_conn is not None:
                result = self._duckdb_conn.execute(query)
                df = result.fetchdf()
                columns = list(df.columns)
                rows = df.values.tolist()
                self.logger.info(f"Executed query using DuckDB, returned {len(rows)} rows")
                return (columns, rows)
            else:
                self.logger.warning("No query engine available, returning empty result")
                return ([], [])
        except Exception as e:
            self.logger.error(f"Failed to execute SQL query: {str(e)}")
            return ([], [])
    
    def create_dataframe(self, data: Any, schema: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Create a DataFrame from data.
        
        Args:
            data: Data to create DataFrame from (dict, list, numpy array, etc.)
            schema: Schema of the DataFrame (list of {"name": str, "type": str})
            
        Returns:
            pd.DataFrame: Created DataFrame
        """
        self.logger.info("Creating DataFrame")
        
        try:
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                if schema:
                    columns = [s.get("name", f"col_{i}") for i, s in enumerate(schema)]
                    df = pd.DataFrame(data, columns=columns)
                else:
                    df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                if schema:
                    columns = [s.get("name", f"col_{i}") for i, s in enumerate(schema)]
                    df = pd.DataFrame(data, columns=columns)
                else:
                    df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            if schema:
                for col_schema in schema:
                    col_name = col_schema.get("name")
                    col_type = col_schema.get("type", "").lower()
                    if col_name in df.columns:
                        if col_type in ["int", "integer", "int64"]:
                            df[col_name] = df[col_name].astype(np.int64)
                        elif col_type in ["float", "double", "float64"]:
                            df[col_name] = df[col_name].astype(np.float64)
                        elif col_type in ["str", "string"]:
                            df[col_name] = df[col_name].astype(str)
            
            self.logger.info(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            self.logger.error(f"Failed to create DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def read_parquet(self, path: str, columns: Optional[List[str]] = None,
                    filters: Optional[List] = None, use_statistics: bool = True) -> pd.DataFrame:
        """
        Read a Parquet file into a DataFrame.
        
        Args:
            path: Path to the Parquet file
            columns: Columns to read (projection pushdown)
            filters: Filters to apply (predicate pushdown)
            use_statistics: Whether to use Parquet statistics for optimization
            
        Returns:
            pd.DataFrame: DataFrame with the data
        """
        self.logger.info(f"Reading Parquet file from {path}")
        
        try:
            import pyarrow.parquet as pq
            
            if os.path.exists(path):
                table = pq.read_table(path, columns=columns)
                df = table.to_pandas()
                
                if filters:
                    for filter_tuple in filters:
                        if len(filter_tuple) == 3:
                            col_name, op, value = filter_tuple
                            if op == '=':
                                df = df[df[col_name] == value]
                            elif op == '>':
                                df = df[df[col_name] > value]
                            elif op == '<':
                                df = df[df[col_name] < value]
                            elif op == '>=':
                                df = df[df[col_name] >= value]
                            elif op == '<=':
                                df = df[df[col_name] <= value]
                            elif op == '!=':
                                df = df[df[col_name] != value]
                
                self.logger.info(f"Successfully read Parquet file from {path}, {len(df)} rows")
                return df
            else:
                self.logger.warning(f"Parquet file not found: {path}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to read Parquet file: {str(e)}")
            return pd.DataFrame()
    
    def read_delta_lake(self, path: str, version: Optional[int] = None,
                       columns: Optional[List[str]] = None, filters: Optional[List] = None) -> pd.DataFrame:
        """
        Read a Delta Lake table into a DataFrame.
        
        Args:
            path: Path to the Delta Lake table
            version: Version of the table to read (None for latest)
            columns: Columns to read (projection pushdown)
            filters: Filters to apply (predicate pushdown)
            
        Returns:
            pd.DataFrame: DataFrame with the data
        """
        version_str = f"version {version}" if version is not None else "latest version"
        self.logger.info(f"Reading Delta Lake table from {path} ({version_str})")
        
        try:
            try:
                from deltalake import DeltaTable
                
                if version is not None:
                    dt = DeltaTable(path, version=version)
                else:
                    dt = DeltaTable(path)
                
                df = dt.to_pandas(columns=columns)
                
                if filters:
                    for filter_tuple in filters:
                        if len(filter_tuple) == 3:
                            col_name, op, value = filter_tuple
                            if col_name in df.columns:
                                if op == '=':
                                    df = df[df[col_name] == value]
                                elif op == '>':
                                    df = df[df[col_name] > value]
                                elif op == '<':
                                    df = df[df[col_name] < value]
                
                self.logger.info(f"Successfully read Delta Lake table, {len(df)} rows")
                return df
            except ImportError:
                delta_path = os.path.join(path, "_delta_log")
                if os.path.exists(delta_path):
                    parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
                    if parquet_files:
                        return self.read_parquet(os.path.join(path, parquet_files[0]), columns, filters)
                
                if os.path.exists(path) and path.endswith('.parquet'):
                    return self.read_parquet(path, columns, filters)
                
                self.logger.warning("deltalake library not available and no parquet fallback found")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to read Delta Lake table: {str(e)}")
            return pd.DataFrame()
    
    def optimize_query(self, query: str) -> str:
        """
        Optimize a SQL query for better performance using basic query rewriting.
        
        Args:
            query: SQL query to optimize
            
        Returns:
            str: Optimized SQL query
        """
        self.logger.info(f"Optimizing SQL query")
        
        try:
            import re
            optimized = query
            
            optimized = re.sub(r'\bSELECT\s+\*\s+FROM', 'SELECT /* consider specifying columns */ * FROM', optimized)
            
            if 'WHERE' in optimized.upper() and 'ORDER BY' in optimized.upper():
                pass
            
            optimized = re.sub(r'\s+', ' ', optimized).strip()
            
            if 'LIKE' in optimized.upper():
                optimized = optimized.replace("LIKE '%", "LIKE '/* consider index */ %")
            
            self.logger.info("Successfully optimized SQL query")
            return optimized
        except Exception as e:
            self.logger.error(f"Failed to optimize query: {str(e)}")
            return query
    
    def explain_query(self, query: str) -> str:
        """
        Explain the execution plan for a SQL query.
        
        Args:
            query: SQL query to explain
            
        Returns:
            str: Explanation of the execution plan
        """
        self.logger.info(f"Explaining SQL query")
        
        try:
            if DATAFUSION_AVAILABLE and self._ctx is not None:
                explain_query = f"EXPLAIN {query}"
                result = self._ctx.sql(explain_query)
                plan_df = result.to_pandas()
                explanation = plan_df.to_string()
                self.logger.info("Generated execution plan using DataFusion")
                return explanation
            elif DUCKDB_AVAILABLE and self._duckdb_conn is not None:
                explain_query = f"EXPLAIN {query}"
                result = self._duckdb_conn.execute(explain_query)
                plan_df = result.fetchdf()
                explanation = plan_df.to_string()
                self.logger.info("Generated execution plan using DuckDB")
                return explanation
            else:
                import re
                explanation_lines = ["Execution Plan Analysis:"]
                
                if re.search(r'\bSELECT\b', query, re.IGNORECASE):
                    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        columns = select_match.group(1).strip()
                        explanation_lines.append(f"  ProjectionExec: columns=[{columns}]")
                
                if re.search(r'\bWHERE\b', query, re.IGNORECASE):
                    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        condition = where_match.group(1).strip()
                        explanation_lines.append(f"    FilterExec: predicate=[{condition}]")
                
                if re.search(r'\bJOIN\b', query, re.IGNORECASE):
                    explanation_lines.append("      JoinExec: type=inner")
                
                if re.search(r'\bGROUP BY\b', query, re.IGNORECASE):
                    explanation_lines.append("        AggregateExec: mode=partial")
                
                if re.search(r'\bORDER BY\b', query, re.IGNORECASE):
                    explanation_lines.append("          SortExec")
                
                from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                if from_match:
                    table_name = from_match.group(1)
                    explanation_lines.append(f"            TableScanExec: table={table_name}")
                
                self.logger.info("Generated basic execution plan analysis")
                return "\n".join(explanation_lines)
        except Exception as e:
            self.logger.error(f"Failed to explain query: {str(e)}")
            return f"Error generating execution plan: {str(e)}"
    
    def execute_geospatial_query(self, query: str,
                                 bounds: Optional[Dict[str, float]] = None,
                                 lat_column: str = "lat", lon_column: str = "lon") -> Tuple[List[str], List[List[Any]]]:
        """
        Execute a SQL query with geospatial functions.

        Args:
            query: SQL query to execute, or the name of a registered table when
                ``bounds`` is provided
            bounds: Optional bounding box filter with keys min_lat, max_lat,
                min_lon, max_lon. When provided, a bounding-box query is built
                against the table named by ``query``.
            lat_column: Name of the latitude column for bounds filtering
            lon_column: Name of the longitude column for bounds filtering

        Returns:
            Tuple[List[str], List[List[Any]]]: Column names and rows of the result
        """
        if bounds is not None:
            min_lat = bounds.get("min_lat", -90.0)
            max_lat = bounds.get("max_lat", 90.0)
            min_lon = bounds.get("min_lon", -180.0)
            max_lon = bounds.get("max_lon", 180.0)
            query = (
                f"SELECT * FROM {query} "
                f"WHERE {lat_column} BETWEEN {min_lat} AND {max_lat} "
                f"AND {lon_column} BETWEEN {min_lon} AND {max_lon}"
            )
            self.logger.info(f"Built bounding-box query from bounds: {bounds}")

        if not self.config.enable_geospatial:
            self.logger.warning("Geospatial functions are not enabled in the configuration")
            return self.execute_sql(query)
        
        self.logger.info(f"Executing geospatial SQL query")
        
        try:
            if DUCKDB_AVAILABLE and self._duckdb_conn is not None:
                try:
                    self._duckdb_conn.execute("INSTALL spatial; LOAD spatial;")
                except:
                    pass
                
                result = self._duckdb_conn.execute(query)
                df = result.fetchdf()
                columns = list(df.columns)
                rows = df.values.tolist()
                self.logger.info(f"Executed geospatial query using DuckDB, returned {len(rows)} rows")
                return (columns, rows)
            elif DATAFUSION_AVAILABLE and self._ctx is not None:
                result = self._ctx.sql(query)
                df = result.to_pandas()
                columns = list(df.columns)
                rows = df.values.tolist()
                self.logger.info(f"Executed geospatial query using DataFusion, returned {len(rows)} rows")
                return (columns, rows)
            else:
                return self.execute_sql(query)
        except Exception as e:
            self.logger.error(f"Failed to execute geospatial query: {str(e)}")
            return self.execute_sql(query)
    
    def join_spatial(self, left_df: Any, right_df: Any, left_geometry: str, right_geometry: str,
                    predicate: str = "intersects", distance: Optional[float] = None) -> pd.DataFrame:
        """
        Perform a spatial join between two DataFrames.
        
        Args:
            left_df: Left DataFrame (pandas or GeoDataFrame)
            right_df: Right DataFrame (pandas or GeoDataFrame)
            left_geometry: Geometry column in left DataFrame
            right_geometry: Geometry column in right DataFrame
            predicate: Spatial predicate (intersects, contains, within, etc.)
            distance: Distance for distance-based predicates (e.g., dwithin)
            
        Returns:
            pd.DataFrame: Joined DataFrame
        """
        if not self.config.enable_geospatial:
            self.logger.warning("Geospatial functions are not enabled in the configuration")
            return pd.DataFrame()
        
        distance_str = f" with distance {distance}" if distance is not None else ""
        self.logger.info(f"Performing spatial join with predicate '{predicate}'{distance_str}")
        
        try:
            try:
                import geopandas as gpd
                
                if not isinstance(left_df, gpd.GeoDataFrame):
                    if left_geometry in left_df.columns:
                        left_gdf = gpd.GeoDataFrame(left_df, geometry=left_geometry)
                    else:
                        left_gdf = gpd.GeoDataFrame(left_df)
                else:
                    left_gdf = left_df
                
                if not isinstance(right_df, gpd.GeoDataFrame):
                    if right_geometry in right_df.columns:
                        right_gdf = gpd.GeoDataFrame(right_df, geometry=right_geometry)
                    else:
                        right_gdf = gpd.GeoDataFrame(right_df)
                else:
                    right_gdf = right_df
                
                if distance is not None and predicate == "dwithin":
                    right_buffered = right_gdf.copy()
                    right_buffered['geometry'] = right_gdf.geometry.buffer(distance)
                    result = gpd.sjoin(left_gdf, right_buffered, how='inner', predicate='intersects')
                else:
                    result = gpd.sjoin(left_gdf, right_gdf, how='inner', predicate=predicate)
                
                self.logger.info(f"Spatial join completed with {len(result)} results using geopandas")
                return result
            except ImportError:
                if isinstance(left_df, pd.DataFrame) and isinstance(right_df, pd.DataFrame):
                    common_cols = set(left_df.columns) & set(right_df.columns)
                    if common_cols:
                        merge_col = list(common_cols - {left_geometry, right_geometry})[0] if len(common_cols) > 2 else list(common_cols)[0]
                        result = pd.merge(left_df, right_df, on=merge_col, how='inner', suffixes=('_left', '_right'))
                        self.logger.info(f"Performed non-spatial merge with {len(result)} results (geopandas not available)")
                        return result
                
                self.logger.warning("geopandas not available and no common columns for merge")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to perform spatial join: {str(e)}")
            return pd.DataFrame()
    
    def spatial_aggregate(self, df: Any, geometry_column: str, group_by: List[str],
                         aggregations: List[Dict]) -> pd.DataFrame:
        """
        Perform spatial aggregations on a DataFrame.
        
        Args:
            df: DataFrame to aggregate (pandas or GeoDataFrame)
            geometry_column: Geometry column to use for spatial aggregations
            group_by: Columns to group by
            aggregations: List of aggregation operations ({"type": str, "column": str, "alias": str})
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        if not self.config.enable_geospatial:
            self.logger.warning("Geospatial functions are not enabled in the configuration")
            return pd.DataFrame()
        
        self.logger.info(f"Performing spatial aggregations on column {geometry_column}")
        
        try:
            if isinstance(df, pd.DataFrame):
                work_df = df.copy()
            else:
                work_df = pd.DataFrame(df)
            
            agg_dict = {}
            for agg in aggregations:
                agg_type = agg.get("type", "").lower()
                agg_column = agg.get("column", "")
                agg_alias = agg.get("alias", f"{agg_type}_{agg_column}")
                
                if agg_column == "*":
                    if agg_type == "count":
                        agg_dict[group_by[0] if group_by else work_df.columns[0]] = (agg_alias, 'count')
                    continue
                
                if agg_column not in work_df.columns:
                    continue
                
                if agg_type == "count":
                    agg_dict[agg_column] = (agg_alias, 'count')
                elif agg_type == "sum":
                    agg_dict[agg_column] = (agg_alias, 'sum')
                elif agg_type == "avg" or agg_type == "mean":
                    agg_dict[agg_column] = (agg_alias, 'mean')
                elif agg_type == "min":
                    agg_dict[agg_column] = (agg_alias, 'min')
                elif agg_type == "max":
                    agg_dict[agg_column] = (agg_alias, 'max')
                elif agg_type == "std":
                    agg_dict[agg_column] = (agg_alias, 'std')
            
            if group_by:
                valid_group_by = [col for col in group_by if col in work_df.columns]
                if valid_group_by:
                    grouped = work_df.groupby(valid_group_by)
                    
                    agg_funcs = {}
                    rename_map = {}
                    for col, (alias, func) in agg_dict.items():
                        if col in work_df.columns:
                            agg_funcs[col] = func
                            rename_map[col] = alias
                    
                    if agg_funcs:
                        result = grouped.agg(agg_funcs).reset_index()
                        result = result.rename(columns=rename_map)
                    else:
                        result = grouped.size().reset_index(name='count')
                else:
                    result = work_df
            else:
                agg_results = {}
                for col, (alias, func) in agg_dict.items():
                    if col in work_df.columns:
                        if func == 'count':
                            agg_results[alias] = work_df[col].count()
                        elif func == 'sum':
                            agg_results[alias] = work_df[col].sum()
                        elif func == 'mean':
                            agg_results[alias] = work_df[col].mean()
                        elif func == 'min':
                            agg_results[alias] = work_df[col].min()
                        elif func == 'max':
                            agg_results[alias] = work_df[col].max()
                        elif func == 'std':
                            agg_results[alias] = work_df[col].std()
                result = pd.DataFrame([agg_results])
            
            self.logger.info(f"Spatial aggregation completed with {len(result)} results")
            return result
        except Exception as e:
            self.logger.error(f"Failed to perform spatial aggregation: {str(e)}")
            return pd.DataFrame()
    
    def close(self) -> None:
        """Close the DataFusion query engine."""
        self.logger.info("Closing DataFusion query engine")
        
        try:
            if self._duckdb_conn is not None:
                self._duckdb_conn.close()
                self._duckdb_conn = None
            self._ctx = None
            self._tables.clear()
            self.logger.info("Successfully closed DataFusion query engine")
        except Exception as e:
            self.logger.error(f"Error closing query engine: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = DataFusionConfig(
        batch_size=8192,
        concurrency=4,
        parquet_pruning=True,
        repartition_joins=True,
        enable_geospatial=True,
        spatial_partitioning=True
    )
    
    # Create a query engine
    engine = DataFusionEngine(config)
    
    # Register tables
    engine.register_table(
        name="satellite_imagery",
        path="/data/mineralvision/lakehouse/processed/satellite_imagery_processed",
        format_type="parquet",
        options={
            "partitioning": "hive"
        }
    )
    
    engine.register_table(
        name="lidar_data",
        path="/data/mineralvision/lakehouse/processed/lidar_features",
        format_type="parquet"
    )
    
    engine.register_table(
        name="geological_features",
        path="/data/mineralvision/lakehouse/processed/geological_features",
        format_type="delta"
    )
    
    # Execute a SQL query
    columns, rows = engine.execute_sql("""
        SELECT
            s.id,
            s.acquisition_date,
            s.sensor_type,
            s.tile_id,
            AVG(s.ndvi) AS ndvi_mean,
            STDDEV(s.ndvi) AS ndvi_std
        FROM
            satellite_imagery s
        WHERE
            s.sensor_type = 'Landsat-8'
            AND s.acquisition_date >= '2025-01-01'
        GROUP BY
            s.id, s.acquisition_date, s.sensor_type, s.tile_id
        ORDER BY
            s.acquisition_date DESC, s.tile_id
    """)
    
    # Execute a geospatial query
    geo_columns, geo_rows = engine.execute_geospatial_query("""
        SELECT
            g.id,
            g.geometry,
            ST_Area(g.geometry) AS area,
            ST_Perimeter(g.geometry) AS perimeter,
            ST_X(ST_Centroid(g.geometry)) AS centroid_x,
            ST_Y(ST_Centroid(g.geometry)) AS centroid_y
        FROM
            geological_features g
        WHERE
            ST_Intersects(g.geometry, ST_GeomFromText('POLYGON((120 60, 130 60, 130 70, 120 70, 120 60))'))
            AND g.feature_type = 'fault'
        ORDER BY
            area DESC
    """)
    
    # Read a Parquet file
    df = engine.read_parquet(
        path="/data/mineralvision/lakehouse/processed/satellite_imagery_processed",
        columns=["id", "acquisition_date", "sensor_type", "tile_id", "ndvi"],
        filters=[("sensor_type", "=", "Landsat-8")],
        use_statistics=True
    )
    
    # Read a Delta Lake table
    delta_df = engine.read_delta_lake(
        path="/data/mineralvision/lakehouse/processed/geological_features",
        version=None,  # Latest version
        columns=["id", "geometry", "feature_type", "confidence"],
        filters=[("feature_type", "=", "fault")]
    )
    
    # Optimize a query
    optimized_query = engine.optimize_query("""
        SELECT
            s.id,
            s.acquisition_date,
            s.sensor_type,
            s.tile_id,
            l.elevation_mean,
            l.slope_mean,
            l.aspect_mean
        FROM
            satellite_imagery s
        JOIN
            lidar_data l
        ON
            s.tile_id = l.tile_id
        WHERE
            s.sensor_type = 'Landsat-8'
            AND s.acquisition_date >= '2025-01-01'
        ORDER BY
            s.acquisition_date DESC, s.tile_id
    """)
    
    # Explain a query
    explanation = engine.explain_query("""
        SELECT
            s.id,
            s.acquisition_date,
            s.sensor_type,
            s.tile_id,
            AVG(s.ndvi) AS ndvi_mean,
            STDDEV(s.ndvi) AS ndvi_std
        FROM
            satellite_imagery s
        WHERE
            s.sensor_type = 'Landsat-8'
            AND s.acquisition_date >= '2025-01-01'
        GROUP BY
            s.id, s.acquisition_date, s.sensor_type, s.tile_id
        ORDER BY
            s.acquisition_date DESC, s.tile_id
    """)
    
    # Perform a spatial join
    joined_df = engine.join_spatial(
        left_df=df,
        right_df=delta_df,
        left_geometry="geometry",
        right_geometry="geometry",
        predicate="intersects"
    )
    
    # Perform spatial aggregations
    aggregated_df = engine.spatial_aggregate(
        df=delta_df,
        geometry_column="geometry",
        group_by=["feature_type"],
        aggregations=[
            {"type": "count", "column": "*", "alias": "count"},
            {"type": "sum", "column": "area", "alias": "total_area"},
            {"type": "avg", "column": "confidence", "alias": "avg_confidence"}
        ]
    )
    
    # Close the engine
    engine.close()
