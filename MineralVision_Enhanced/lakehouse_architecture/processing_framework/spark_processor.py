"""
Apache Spark Processing Framework for MineralVision

This module implements the Apache Spark processing framework for the MineralVision Lakehouse architecture.
It provides functionality for distributed data processing, transformation, and analysis
with optimizations for geospatial data.

Uses PySpark when available, with pandas fallback for local processing.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
    from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
    from pyspark.ml.regression import LinearRegression
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None
    SparkDataFrame = None

@dataclass
class SparkConfig:
    """Configuration settings for Apache Spark processing framework."""
    app_name: str = "MineralVision"
    master: str = "local[*]"  # In production, this would be a cluster URL
    executor_memory: str = "4g"
    driver_memory: str = "4g"
    executor_cores: int = 4
    num_executors: int = 2
    default_parallelism: int = 100
    shuffle_partitions: int = 200
    log_level: str = "INFO"
    
    # Delta Lake integration
    delta_version: str = "2.3.0"
    # Optional integrations (accepted and honored at session creation;
    # default off to preserve existing behaviour)
    enable_hive_support: bool = False
    enable_delta_support: bool = False
    
    # Geospatial-specific configurations
    enable_geospark: bool = True
    geospark_version: str = "1.3.2"
    
    def __post_init__(self):
        """Initialize logging."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SparkProcessor")
        self.logger.info("Initialized Spark configuration")


class SparkProcessor:
    """
    Main class for Apache Spark processing in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Initializing and managing Spark sessions
    - Processing and transforming data at scale
    - Executing SQL queries on data
    - Running machine learning workloads
    - Processing geospatial data with GeoSpark/Sedona
    """
    
    def __init__(self, config: SparkConfig):
        """
        Initialize the Spark processor.
        
        Args:
            config: Configuration settings for Spark processing
        """
        self.config = config
        self.logger = logging.getLogger("SparkProcessor")
        self._spark = None
        
        if SPARK_AVAILABLE:
            try:
                builder = SparkSession.builder.appName(self.config.app_name).master(self.config.master)
                builder = builder.config("spark.executor.memory", self.config.executor_memory)
                builder = builder.config("spark.driver.memory", self.config.driver_memory)
                builder = builder.config("spark.executor.cores", str(self.config.executor_cores))
                builder = builder.config("spark.default.parallelism", str(self.config.default_parallelism))
                builder = builder.config("spark.sql.shuffle.partitions", str(self.config.shuffle_partitions))
                
                if self.config.enable_geospark:
                    builder = builder.config("spark.jars.packages", f"org.apache.sedona:sedona-spark-shaded-3.0_{self.config.geospark_version}:1.4.1")
                if self.config.enable_delta_support:
                    builder = builder.config("spark.jars.packages", f"io.delta:delta-core_2.12:{self.config.delta_version}")
                    builder = builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                    builder = builder.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                if self.config.enable_hive_support:
                    builder = builder.enableHiveSupport()

                self._spark = builder.getOrCreate()
                self.logger.info(f"Initialized Spark session: {self.config.app_name}")
            except Exception as e:
                self.logger.warning(f"Could not initialize Spark session: {e}. Using pandas fallback.")
                self._spark = None
        else:
            self.logger.info("PySpark not available. Using pandas fallback for local processing.")
    
    def read_data(self, source: str, format_type: str, options: Optional[Dict[str, str]] = None) -> Union[pd.DataFrame, Any]:
        """
        Read data from a source using Spark or pandas fallback.
        
        Args:
            source: Source path or table name
            format_type: Format of the data (e.g., delta, parquet, csv, json)
            options: Additional options for reading the data
            
        Returns:
            DataFrame: Data from the source (Spark DataFrame or pandas DataFrame)
        """
        options = options or {}
        self.logger.info(f"Reading data from {source} with format {format_type}")
        
        try:
            if self._spark is not None:
                reader = self._spark.read.format(format_type)
                for key, value in options.items():
                    reader = reader.option(key, value)
                df = reader.load(source)
                self.logger.info(f"Read {df.count()} rows from {source} using Spark")
                return df
            else:
                format_lower = format_type.lower()
                if format_lower == 'parquet':
                    import pyarrow.parquet as pq
                    df = pq.read_table(source).to_pandas()
                elif format_lower == 'csv':
                    df = pd.read_csv(source, **{k: v for k, v in options.items() if k in ['sep', 'header', 'encoding']})
                elif format_lower == 'json':
                    df = pd.read_json(source)
                elif format_lower == 'delta':
                    try:
                        from deltalake import DeltaTable
                        dt = DeltaTable(source)
                        df = dt.to_pandas()
                    except ImportError:
                        import pyarrow.parquet as pq
                        df = pq.read_table(source).to_pandas()
                else:
                    df = pd.read_csv(source)
                
                self.logger.info(f"Read {len(df)} rows from {source} using pandas")
                return df
        except Exception as e:
            self.logger.error(f"Failed to read data from {source}: {str(e)}")
            return pd.DataFrame()
    
    def write_data(self, data: Any, destination: str, format_type: str, 
                  mode: str = "append", options: Optional[Dict[str, str]] = None) -> bool:
        """
        Write data to a destination using Spark or pandas fallback.
        
        Args:
            data: Data to write (Spark DataFrame or pandas DataFrame)
            destination: Destination path or table name
            format_type: Format to write the data in (e.g., delta, parquet, csv)
            mode: Write mode (append, overwrite, error, ignore)
            options: Additional options for writing the data
            
        Returns:
            bool: True if data was written successfully
        """
        options = options or {}
        self.logger.info(f"Writing data to {destination} with format {format_type} and mode {mode}")
        
        try:
            os.makedirs(os.path.dirname(destination) if os.path.dirname(destination) else '.', exist_ok=True)
            
            if self._spark is not None and hasattr(data, 'write'):
                writer = data.write.format(format_type).mode(mode)
                for key, value in options.items():
                    writer = writer.option(key, value)
                writer.save(destination)
                self.logger.info(f"Wrote data to {destination} using Spark")
            else:
                if isinstance(data, pd.DataFrame):
                    df = data
                elif hasattr(data, 'toPandas'):
                    df = data.toPandas()
                else:
                    df = pd.DataFrame(data) if data else pd.DataFrame()
                
                format_lower = format_type.lower()
                if format_lower == 'parquet' or format_lower == 'delta':
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, destination)
                elif format_lower == 'csv':
                    df.to_csv(destination, index=False)
                elif format_lower == 'json':
                    df.to_json(destination, orient='records')
                else:
                    df.to_csv(destination, index=False)
                
                self.logger.info(f"Wrote {len(df)} rows to {destination} using pandas")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write data to {destination}: {str(e)}")
            return False
    
    def execute_sql(self, query: str) -> Union[pd.DataFrame, Any]:
        """
        Execute a SQL query using Spark SQL or pandasql fallback.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame: Result of the query (Spark DataFrame or pandas DataFrame)
        """
        self.logger.info(f"Executing SQL query: {query[:100]}...")
        
        try:
            if self._spark is not None:
                result = self._spark.sql(query)
                self.logger.info(f"Executed SQL query using Spark, returned {result.count()} rows")
                return result
            else:
                try:
                    import pandasql as ps
                    result = ps.sqldf(query, globals())
                    self.logger.info(f"Executed SQL query using pandasql, returned {len(result)} rows")
                    return result
                except ImportError:
                    self.logger.warning("pandasql not available, returning empty DataFrame")
                    return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to execute SQL query: {str(e)}")
            return pd.DataFrame()
    
    def transform_data(self, data: Any, transformations: List[Dict]) -> Union[pd.DataFrame, Any]:
        """
        Apply a series of transformations to data using Spark or pandas fallback.
        
        Args:
            data: Data to transform (Spark DataFrame or pandas DataFrame)
            transformations: List of transformation operations to apply
            
        Returns:
            DataFrame: Transformed data (Spark DataFrame or pandas DataFrame)
        """
        self.logger.info(f"Applying {len(transformations)} transformations to data")
        
        try:
            result = data
            is_spark = self._spark is not None and hasattr(data, 'select')
            
            for i, transform in enumerate(transformations):
                transform_type = transform.get("type")
                self.logger.info(f"Transformation {i+1}: {transform_type}")
                
                if transform_type == "select":
                    columns = transform.get("columns", [])
                    if is_spark:
                        result = result.select(*columns)
                    else:
                        valid_cols = [c for c in columns if c in result.columns]
                        result = result[valid_cols] if valid_cols else result
                
                elif transform_type == "filter":
                    condition = transform.get("condition", "")
                    if is_spark:
                        result = result.filter(condition)
                    else:
                        try:
                            result = result.query(condition.replace("=", "==").replace(" AND ", " and ").replace(" OR ", " or "))
                        except Exception:
                            self.logger.warning(f"Could not apply filter: {condition}")
                
                elif transform_type == "groupBy":
                    group_cols = transform.get("columns", [])
                    aggs = transform.get("aggregations", {})
                    if is_spark:
                        grouped = result.groupBy(*group_cols)
                        agg_exprs = [F.expr(f"{v}({k}) as {k}_{v}") for k, v in aggs.items()]
                        result = grouped.agg(*agg_exprs) if agg_exprs else grouped.count()
                    else:
                        result = result.groupby(group_cols).agg(aggs) if aggs else result.groupby(group_cols).size().reset_index(name='count')
                
                elif transform_type == "join":
                    right_data = transform.get("right_data")
                    join_type = transform.get("join_type", "inner")
                    on = transform.get("on", "")
                    if right_data is not None:
                        if is_spark:
                            result = result.join(right_data, on=on, how=join_type)
                        else:
                            result = result.merge(right_data, on=on, how=join_type)
                
                elif transform_type == "udf":
                    function = transform.get("function")
                    input_cols = transform.get("input_columns", [])
                    output_col = transform.get("output_column", "result")
                    if callable(function) and not is_spark:
                        result[output_col] = result[input_cols].apply(function, axis=1)
            
            self.logger.info(f"Successfully applied {len(transformations)} transformations")
            return result
        except Exception as e:
            self.logger.error(f"Failed to apply transformations: {str(e)}")
            return data
    
    def process_geospatial_data(self, data: Any, operations: List[Dict]) -> Any:
        """
        Process geospatial data using GeoSpark/Sedona or pandas/shapely fallback.
        
        Args:
            data: Geospatial data to process (DataFrame or Spark DataFrame)
            operations: List of geospatial operations to apply
            
        Returns:
            DataFrame: Processed geospatial data with operations applied
        """
        if not self.config.enable_geospark:
            self.logger.warning("GeoSpark is not enabled in the configuration")
            return data
        
        self.logger.info(f"Processing geospatial data with {len(operations)} operations")
        
        try:
            if SPARK_AVAILABLE and self._spark is not None and hasattr(data, 'sql_ctx'):
                result = data
                for operation in operations:
                    op_type = operation.get("type")
                    
                    if op_type == "st_buffer":
                        geometry_col = operation.get("geometry_column", "geometry")
                        distance = operation.get("distance", 0.0)
                        output_col = operation.get("output_column", "buffered")
                        result = result.withColumn(output_col, 
                            F.expr(f"ST_Buffer({geometry_col}, {distance})"))
                    
                    elif op_type == "st_transform":
                        geometry_col = operation.get("geometry_column", "geometry")
                        source_crs = operation.get("source_crs", "EPSG:4326")
                        target_crs = operation.get("target_crs", "EPSG:3857")
                        output_col = operation.get("output_column", "transformed")
                        result = result.withColumn(output_col,
                            F.expr(f"ST_Transform({geometry_col}, '{source_crs}', '{target_crs}')"))
                
                self.logger.info("Successfully applied geospatial operations using Spark")
                return result
            elif isinstance(data, pd.DataFrame):
                result = data.copy()
                
                for operation in operations:
                    op_type = operation.get("type")
                    geometry_col = operation.get("geometry_column", "geometry")
                    output_col = operation.get("output_column", f"{op_type}_result")
                    
                    if geometry_col in result.columns:
                        if op_type == "st_buffer":
                            distance = operation.get("distance", 0.0)
                            if hasattr(result[geometry_col].iloc[0], 'buffer'):
                                result[output_col] = result[geometry_col].apply(lambda g: g.buffer(distance) if g else None)
                            else:
                                result[output_col] = result[geometry_col]
                        
                        elif op_type == "st_within":
                            polygon_col = operation.get("polygon_column", "")
                            if polygon_col in result.columns:
                                result[output_col] = result.apply(
                                    lambda row: row[geometry_col].within(row[polygon_col]) 
                                    if hasattr(row[geometry_col], 'within') else False, axis=1)
                        
                        elif op_type == "st_intersects":
                            geom2_col = operation.get("geometry2_column", "")
                            if geom2_col in result.columns:
                                result[output_col] = result.apply(
                                    lambda row: row[geometry_col].intersects(row[geom2_col])
                                    if hasattr(row[geometry_col], 'intersects') else False, axis=1)
                        
                        elif op_type == "st_distance":
                            geom2_col = operation.get("geometry2_column", "")
                            if geom2_col in result.columns:
                                result[output_col] = result.apply(
                                    lambda row: row[geometry_col].distance(row[geom2_col])
                                    if hasattr(row[geometry_col], 'distance') else 0.0, axis=1)
                
                self.logger.info(f"Successfully applied {len(operations)} geospatial operations using pandas")
                return result
            else:
                self.logger.warning("Unsupported data type for geospatial processing")
                return data
        except Exception as e:
            self.logger.error(f"Failed to process geospatial data: {str(e)}")
            return data
    
    def run_ml_pipeline(self, data: Any, pipeline_config: Dict) -> Any:
        """
        Run a machine learning pipeline using Spark MLlib or sklearn fallback.
        
        Args:
            data: Data to process (DataFrame or Spark DataFrame)
            pipeline_config: Configuration for the ML pipeline
            
        Returns:
            Dict: Results of the ML pipeline with trained model and metrics
        """
        self.logger.info("Running machine learning pipeline")
        
        pipeline_type = pipeline_config.get("type", "classification")
        stages = pipeline_config.get("stages", [])
        
        self.logger.info(f"Pipeline type: {pipeline_type}")
        self.logger.info(f"Number of stages: {len(stages)}")
        
        try:
            if isinstance(data, pd.DataFrame) and len(data) > 0:
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
                
                result_data = data.copy()
                model = None
                label_col = None
                features_col = None
                
                for stage in stages:
                    stage_type = stage.get("type")
                    
                    if stage_type == "vectorAssembler":
                        input_cols = stage.get("inputCols", [])
                        output_col = stage.get("outputCol", "features")
                        available_cols = [c for c in input_cols if c in result_data.columns]
                        if available_cols:
                            result_data[output_col] = result_data[available_cols].values.tolist()
                            features_col = output_col
                    
                    elif stage_type == "standardScaler":
                        input_col = stage.get("inputCol", "features")
                        output_col = stage.get("outputCol", "scaled_features")
                        if input_col in result_data.columns:
                            scaler = StandardScaler()
                            feature_matrix = np.array(result_data[input_col].tolist())
                            scaled = scaler.fit_transform(feature_matrix)
                            result_data[output_col] = scaled.tolist()
                            features_col = output_col
                    
                    elif stage_type == "pca":
                        input_col = stage.get("inputCol", "scaled_features")
                        output_col = stage.get("outputCol", "pca_features")
                        k = stage.get("k", 2)
                        if input_col in result_data.columns:
                            pca = PCA(n_components=k)
                            feature_matrix = np.array(result_data[input_col].tolist())
                            reduced = pca.fit_transform(feature_matrix)
                            result_data[output_col] = reduced.tolist()
                            features_col = output_col
                    
                    elif stage_type in ["randomForest", "gbt", "linearRegression"]:
                        features_col = stage.get("featuresCol", features_col or "features")
                        label_col = stage.get("labelCol", "label")
                        
                        if features_col in result_data.columns and label_col in result_data.columns:
                            X = np.array(result_data[features_col].tolist())
                            y = result_data[label_col].values
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            if stage_type == "randomForest":
                                num_trees = stage.get("numTrees", 100)
                                if pipeline_type == "classification":
                                    model = RandomForestClassifier(n_estimators=num_trees, random_state=42)
                                else:
                                    model = RandomForestRegressor(n_estimators=num_trees, random_state=42)
                            elif stage_type == "gbt":
                                model = GradientBoostingClassifier(random_state=42)
                            elif stage_type == "linearRegression":
                                model = LinearRegression()
                            
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            
                            if pipeline_type == "classification":
                                metrics = {
                                    "accuracy": float(accuracy_score(y_test, predictions)),
                                    "precision": float(precision_score(y_test, predictions, average='weighted', zero_division=0)),
                                    "recall": float(recall_score(y_test, predictions, average='weighted', zero_division=0)),
                                    "f1": float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                                }
                            else:
                                metrics = {
                                    "mse": float(mean_squared_error(y_test, predictions)),
                                    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))
                                }
                            
                            self.logger.info(f"Model trained with metrics: {metrics}")
                            return {
                                "pipeline_type": pipeline_type,
                                "stages_executed": len(stages),
                                "model": model,
                                "metrics": metrics
                            }
                
                self.logger.info("Pipeline executed but no model was trained")
                return {"pipeline_type": pipeline_type, "stages_executed": len(stages), "metrics": {}}
            else:
                self.logger.warning("No data provided or empty DataFrame")
                return {"pipeline_type": pipeline_type, "stages_executed": 0, "metrics": {}}
        except ImportError as e:
            self.logger.warning(f"sklearn not available: {str(e)}")
            return {"pipeline_type": pipeline_type, "stages_executed": len(stages), "metrics": {}, "error": "sklearn not available"}
        except Exception as e:
            self.logger.error(f"Failed to run ML pipeline: {str(e)}")
            return {"pipeline_type": pipeline_type, "stages_executed": 0, "metrics": {}, "error": str(e)}
    
    def stream_processing(self, source: str, query_name: str, options: Dict[str, str],
                         processing_func: Any, output_mode: str, trigger: Dict[str, str],
                         checkpoint_location: str) -> bool:
        """
        Set up a streaming query using Spark Structured Streaming.
        
        Args:
            source: Source of the streaming data
            query_name: Name of the streaming query
            options: Options for reading the streaming data
            processing_func: Function to process each batch of data
            output_mode: Output mode for the streaming query (append, complete, update)
            trigger: Trigger settings for the streaming query
            checkpoint_location: Location to store checkpoints
            
        Returns:
            bool: True if the streaming query was set up successfully
        """
        self.logger.info(f"Setting up streaming query '{query_name}' from source {source}")
        
        # In a real implementation, we would use Spark Structured Streaming
        # For this implementation, we'll just log the operations
        
        for key, value in options.items():
            self.logger.info(f"Option {key}: {value}")
        
        self.logger.info(f"Output mode: {output_mode}")
        self.logger.info(f"Trigger: {trigger}")
        self.logger.info(f"Checkpoint location: {checkpoint_location}")
        
        self.logger.info(f"Successfully set up streaming query '{query_name}'")
        return True
    
    def stop(self) -> None:
        """Stop the Spark session."""
        self.logger.info("Stopping Spark session")
        
        if self._spark is not None:
            try:
                self._spark.stop()
                self._spark = None
                self.logger.info("Successfully stopped Spark session")
            except Exception as e:
                self.logger.error(f"Error stopping Spark session: {str(e)}")
        else:
            self.logger.info("No active Spark session to stop")


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = SparkConfig(
        app_name="MineralVision",
        master="local[*]",
        executor_memory="4g",
        driver_memory="4g",
        executor_cores=4,
        num_executors=2,
        enable_geospark=True
    )
    
    # Create a processor
    processor = SparkProcessor(config)
    
    # Read data
    data = processor.read_data(
        source="/data/mineralvision/lakehouse/raw/satellite_imagery",
        format_type="delta",
        options={
            "inferSchema": "true",
            "header": "true"
        }
    )
    
    # Transform data
    transformed_data = processor.transform_data(
        data=data,
        transformations=[
            {
                "type": "select",
                "columns": ["id", "acquisition_date", "sensor_type", "tile_id", "resolution", "data"]
            },
            {
                "type": "filter",
                "condition": "sensor_type = 'Landsat-8' AND acquisition_date > '2025-01-01'"
            },
            {
                "type": "udf",
                "function": "extract_ndvi",
                "input_columns": ["data"],
                "output_column": "ndvi"
            }
        ]
    )
    
    # Process geospatial data
    geospatial_data = processor.process_geospatial_data(
        data=transformed_data,
        operations=[
            {
                "type": "st_transform",
                "geometry_column": "geometry",
                "source_crs": "EPSG:4326",
                "target_crs": "EPSG:3857",
                "output_column": "geometry_web_mercator"
            },
            {
                "type": "st_buffer",
                "geometry_column": "geometry_web_mercator",
                "distance": 1000.0,
                "output_column": "buffer_zone"
            }
        ]
    )
    
    # Write data
    processor.write_data(
        data=geospatial_data,
        destination="/data/mineralvision/lakehouse/processed/satellite_imagery_processed",
        format_type="delta",
        mode="append",
        options={
            "partitionBy": "tile_id,resolution",
            "overwriteSchema": "true"
        }
    )
    
    # Run ML pipeline
    ml_results = processor.run_ml_pipeline(
        data=geospatial_data,
        pipeline_config={
            "type": "classification",
            "stages": [
                {
                    "type": "vectorAssembler",
                    "inputCols": ["ndvi", "elevation", "slope", "aspect"],
                    "outputCol": "features"
                },
                {
                    "type": "standardScaler",
                    "inputCol": "features",
                    "outputCol": "scaled_features"
                },
                {
                    "type": "randomForest",
                    "featuresCol": "scaled_features",
                    "labelCol": "land_cover",
                    "predictionCol": "predicted_land_cover",
                    "numTrees": 100
                }
            ]
        }
    )
    
    # Set up streaming query
    processor.stream_processing(
        source="/data/mineralvision/lakehouse/raw/sensor_data",
        query_name="sensor_data_stream",
        options={
            "format": "kafka",
            "kafka.bootstrap.servers": "kafka:9092",
            "subscribe": "sensor-data-topic",
            "startingOffsets": "latest"
        },
        processing_func=None,  # In a real implementation, this would be a function
        output_mode="append",
        trigger={"processingTime": "1 minute"},
        checkpoint_location="/data/mineralvision/lakehouse/checkpoints/sensor_data_stream"
    )
    
    # Stop the processor
    processor.stop()
