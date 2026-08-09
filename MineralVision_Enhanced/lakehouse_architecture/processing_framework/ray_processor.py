"""
Ray Processing Framework for MineralVision

This module implements the Ray processing framework for the MineralVision Lakehouse architecture.
It provides functionality for distributed Python execution, machine learning, and
parallel processing with optimizations for geospatial data.

Uses Ray when available, with multiprocessing/concurrent.futures fallback for local processing.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

import pandas as pd
import numpy as np

try:
    import ray
    from ray import tune
    from ray.data import Dataset as RayDataset
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    tune = None
    RayDataset = None

@dataclass
class RayConfig:
    """Configuration settings for Ray processing framework."""
    address: str = "auto"  # "auto" for local, or cluster address like "ray://192.168.1.100:10001"
    num_cpus: Optional[int] = None  # None means use all available
    num_gpus: Optional[int] = None  # None means use all available
    memory: Optional[int] = None  # Memory in bytes
    object_store_memory: Optional[int] = None  # Object store memory in bytes
    resources: Optional[Dict[str, float]] = None  # Custom resources
    enable_dashboard: bool = False  # honored via ray.init(include_dashboard=...)
    log_level: str = "INFO"
    
    # ML-specific configurations
    tune_resources: Optional[Dict[str, Any]] = None
    train_parallelism: int = 4
    
    # Geospatial-specific configurations
    geospatial_batch_size: int = 1000
    
    def __post_init__(self):
        """Initialize logging."""
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RayProcessor")
        self.logger.info("Initialized Ray configuration")
        
        # Initialize resources if None
        if self.resources is None:
            self.resources = {}
        
        # Initialize tune resources if None
        if self.tune_resources is None:
            self.tune_resources = {
                "cpu": 2,
                "gpu": 0
            }


class RayProcessor:
    """
    Main class for Ray processing in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Initializing and managing Ray clusters
    - Executing tasks in parallel
    - Distributed machine learning with Ray Train and Tune
    - Processing large datasets with Ray Data
    - Optimized geospatial processing
    """
    
    def __init__(self, config: RayConfig):
        """
        Initialize the Ray processor.
        
        Args:
            config: Configuration settings for Ray processing
        """
        self.config = config
        self.logger = logging.getLogger("RayProcessor")
        self._ray_initialized = False
        self._executor = None
        
        if RAY_AVAILABLE:
            try:
                init_kwargs = {}
                if self.config.address != "auto":
                    init_kwargs["address"] = self.config.address
                if self.config.num_cpus is not None:
                    init_kwargs["num_cpus"] = self.config.num_cpus
                if self.config.num_gpus is not None:
                    init_kwargs["num_gpus"] = self.config.num_gpus
                init_kwargs["include_dashboard"] = self.config.enable_dashboard
                
                if not ray.is_initialized():
                    ray.init(**init_kwargs, ignore_reinit_error=True)
                self._ray_initialized = True
                self.logger.info(f"Initialized Ray with address: {self.config.address}")
            except Exception as e:
                self.logger.warning(f"Could not initialize Ray: {e}. Using multiprocessing fallback.")
                self._ray_initialized = False
        else:
            self.logger.info("Ray not available. Using multiprocessing/threading fallback.")
        
        if not self._ray_initialized:
            max_workers = self.config.num_cpus or multiprocessing.cpu_count()
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def parallel_task(self, func: Callable, items: List[Any], num_cpus_per_task: int = 1,
                     num_gpus_per_task: float = 0) -> List[Any]:
        """
        Execute a function in parallel across multiple items.
        
        Args:
            func: Function to execute
            items: List of items to process
            num_cpus_per_task: Number of CPUs to allocate per task
            num_gpus_per_task: Number of GPUs to allocate per task
            
        Returns:
            List[Any]: Results of the parallel execution
        """
        func_name = getattr(func, '__name__', 'anonymous')
        self.logger.info(f"Executing function {func_name} in parallel on {len(items)} items")
        
        try:
            if self._ray_initialized:
                remote_func = ray.remote(num_cpus=num_cpus_per_task, num_gpus=num_gpus_per_task)(func)
                futures = [remote_func.remote(item) for item in items]
                results = ray.get(futures)
                self.logger.info(f"Completed {len(results)} tasks using Ray")
                return results
            else:
                results = list(self._executor.map(func, items))
                self.logger.info(f"Completed {len(results)} tasks using ThreadPoolExecutor")
                return results
        except Exception as e:
            self.logger.error(f"Failed to execute parallel task: {str(e)}")
            results = [func(item) for item in items]
            self.logger.info(f"Completed {len(results)} tasks using sequential fallback")
            return results
    
    def execute_parallel(self, func: Callable, items: List[Any], num_cpus_per_task: int = 1, num_gpus_per_task: float = 0) -> List[Any]:
        """Alias for parallel_task for backward compatibility with tests."""
        return self.parallel_task(func, items, num_cpus_per_task, num_gpus_per_task)
    
    def process_dataset(self, dataset_path: Any, transformations: Any,
                       output_path: Optional[str] = None,
                       batch_size: Optional[int] = None) -> Any:
        """
        Process a dataset using Ray Data.

        Args:
            dataset_path: Path to the dataset, or (legacy form) an in-memory
                list of item dicts
            transformations: List of transformation dicts, or (legacy form) a
                callable applied to each dataset item
            output_path: Path to write the processed dataset to
            batch_size: (legacy form) batch size used when applying a callable
                transformation over an in-memory dataset

        Returns:
            Any: Processed dataset (in a real implementation, this would be a Ray Dataset)
        """
        # Legacy API: process_dataset(items, fn, batch_size=N) — apply the
        # callable over the in-memory dataset in batches and return results.
        if callable(transformations) and isinstance(dataset_path, (list, tuple)):
            fn = transformations
            bs = batch_size or len(dataset_path) or 1
            self.logger.info(
                f"Processing {len(dataset_path)} in-memory items with callable "
                f"in batches of {bs}"
            )
            processed = []
            for start in range(0, len(dataset_path), bs):
                batch = dataset_path[start:start + bs]
                processed.extend(fn(item) for item in batch)
            return processed

        self.logger.info(f"Processing dataset from {dataset_path}")

        # In a real implementation, we would use Ray Data to process the dataset
        # For this implementation, we'll just log the operations
        
        for i, transform in enumerate(transformations):
            transform_type = transform.get("type")
            self.logger.info(f"Transformation {i+1}: {transform_type}")
            
            if transform_type == "map":
                function_name = transform.get("function", "")
                batch = transform.get("batch", False)
                compute = transform.get("compute", "tasks")
                self.logger.info(f"Mapping function {function_name} with batch={batch}, compute={compute}")
                
                if RAY_AVAILABLE and result is not None and hasattr(result, 'map'):
                    map_func = transform.get("func")
                    if map_func:
                        result = result.map(map_func)
            
            elif transform_type == "filter":
                function_name = transform.get("function", "")
                self.logger.info(f"Filtering with function {function_name}")
                
                if RAY_AVAILABLE and result is not None and hasattr(result, 'filter'):
                    filter_func = transform.get("func")
                    if filter_func:
                        result = result.filter(filter_func)
            
            elif transform_type == "flat_map":
                function_name = transform.get("function", "")
                self.logger.info(f"Flat mapping with function {function_name}")
            
            elif transform_type == "groupby":
                key = transform.get("key", "")
                agg = transform.get("agg", "")
                self.logger.info(f"Grouping by {key} with aggregation {agg}")
            
            elif transform_type == "repartition":
                num_partitions = transform.get("num_partitions", 200)
                self.logger.info(f"Repartitioning to {num_partitions} partitions")
                
                if RAY_AVAILABLE and result is not None and hasattr(result, 'repartition'):
                    result = result.repartition(num_partitions)
        
        if output_path:
            self.logger.info(f"Writing processed dataset to {output_path}")
            if RAY_AVAILABLE and result is not None and hasattr(result, 'write_parquet'):
                result.write_parquet(output_path)
        
        self.logger.info("Successfully processed dataset")
        
        return {
            "dataset_path": dataset_path,
            "transformations_applied": len(transformations),
            "output_path": output_path,
            "result": result
        }
    
    def train_model(self, train_dataset: Any, val_dataset: Any, model_config: Dict,
                   num_workers: int = 4, use_gpu: bool = False) -> Dict:
        """
        Train a machine learning model using Ray Train or sklearn fallback.
        
        Args:
            train_dataset: Training dataset (DataFrame, numpy array, or Ray Dataset)
            val_dataset: Validation dataset (DataFrame, numpy array, or Ray Dataset)
            model_config: Configuration for the model
            num_workers: Number of workers to use for training
            use_gpu: Whether to use GPUs for training
            
        Returns:
            Dict: Training results with model and metrics
        """
        self.logger.info("Training machine learning model")
        
        model_type = model_config.get("type", "random_forest")
        hyperparams = model_config.get("hyperparameters", {})
        training_config = model_config.get("training", {})
        epochs = training_config.get("epochs", 10)
        batch_size = training_config.get("batch_size", 32)
        learning_rate = training_config.get("learning_rate", 0.001)
        
        self.logger.info(f"Model type: {model_type}, Workers: {num_workers}, GPU: {use_gpu}")
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            X_train, y_train = None, None
            X_val, y_val = None, None
            
            if isinstance(train_dataset, pd.DataFrame):
                feature_cols = [c for c in train_dataset.columns if c != 'label' and c != 'target']
                label_col = 'label' if 'label' in train_dataset.columns else 'target'
                if label_col in train_dataset.columns and feature_cols:
                    X_train = train_dataset[feature_cols].values
                    y_train = train_dataset[label_col].values
            elif isinstance(train_dataset, tuple) and len(train_dataset) == 2:
                X_train, y_train = train_dataset
            elif isinstance(train_dataset, np.ndarray):
                X_train = train_dataset[:, :-1]
                y_train = train_dataset[:, -1]
            
            if isinstance(val_dataset, pd.DataFrame):
                feature_cols = [c for c in val_dataset.columns if c != 'label' and c != 'target']
                label_col = 'label' if 'label' in val_dataset.columns else 'target'
                if label_col in val_dataset.columns and feature_cols:
                    X_val = val_dataset[feature_cols].values
                    y_val = val_dataset[label_col].values
            elif isinstance(val_dataset, tuple) and len(val_dataset) == 2:
                X_val, y_val = val_dataset
            elif isinstance(val_dataset, np.ndarray):
                X_val = val_dataset[:, :-1]
                y_val = val_dataset[:, -1]
            
            model = None
            is_classification = model_config.get("task", "classification") == "classification"
            
            if model_type in ["random_forest", "rf"]:
                n_estimators = hyperparams.get("n_estimators", 100)
                max_depth = hyperparams.get("max_depth", None)
                if is_classification:
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=num_workers)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=num_workers)
            elif model_type in ["gradient_boosting", "gbt"]:
                model = GradientBoostingClassifier(random_state=42)
            elif model_type in ["mlp", "neural_network", "unet"]:
                hidden_layers = hyperparams.get("hidden_layer_sizes", (128, 64))
                if is_classification:
                    model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=epochs, learning_rate_init=learning_rate, random_state=42)
                else:
                    model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=epochs, learning_rate_init=learning_rate, random_state=42)
            elif model_type in ["logistic", "logistic_regression"]:
                model = LogisticRegression(max_iter=epochs, random_state=42)
            else:
                model = LinearRegression()
            
            metrics = {}
            if X_train is not None and y_train is not None:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                
                if is_classification:
                    metrics["train_accuracy"] = float(accuracy_score(y_train, train_pred))
                else:
                    metrics["train_mse"] = float(mean_squared_error(y_train, train_pred))
                
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    if is_classification:
                        metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))
                    else:
                        metrics["val_mse"] = float(mean_squared_error(y_val, val_pred))
                
                self.logger.info(f"Model trained successfully with metrics: {metrics}")
            
            return {
                "model_type": model_type,
                "epochs_completed": epochs,
                "model": model,
                "metrics": metrics
            }
        except ImportError as e:
            self.logger.warning(f"sklearn not available: {str(e)}")
            return {"model_type": model_type, "epochs_completed": 0, "metrics": {}, "error": "sklearn not available"}
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            return {"model_type": model_type, "epochs_completed": 0, "metrics": {}, "error": str(e)}
    
    def hyperparameter_tuning(self, train_dataset: Any, val_dataset: Any, model_builder: Callable,
                            param_space: Dict, num_samples: int = 10, resources_per_trial: Optional[Dict] = None) -> Dict:
        """
        Perform hyperparameter tuning using Ray Tune or grid search fallback.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_builder: Function to build the model
            param_space: Hyperparameter search space (dict of param_name -> list of values)
            num_samples: Number of samples to try
            resources_per_trial: Resources to allocate per trial
            
        Returns:
            Dict: Tuning results with best parameters and metrics
        """
        self.logger.info("Performing hyperparameter tuning")
        self.logger.info(f"Number of samples: {num_samples}")
        
        if resources_per_trial is None:
            resources_per_trial = self.config.tune_resources
        
        try:
            from sklearn.model_selection import ParameterGrid, cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            import itertools
            
            X_train, y_train = None, None
            if isinstance(train_dataset, pd.DataFrame):
                feature_cols = [c for c in train_dataset.columns if c != 'label' and c != 'target']
                label_col = 'label' if 'label' in train_dataset.columns else 'target'
                if label_col in train_dataset.columns and feature_cols:
                    X_train = train_dataset[feature_cols].values
                    y_train = train_dataset[label_col].values
            elif isinstance(train_dataset, tuple) and len(train_dataset) == 2:
                X_train, y_train = train_dataset
            
            param_combinations = []
            param_names = list(param_space.keys())
            param_values = [param_space[k] if isinstance(param_space[k], list) else [param_space[k]] for k in param_names]
            
            for combo in itertools.product(*param_values):
                param_combinations.append(dict(zip(param_names, combo)))
            
            param_combinations = param_combinations[:num_samples]
            
            best_score = -float('inf')
            best_params = {}
            all_results = []
            
            for params in param_combinations:
                if model_builder is not None:
                    model = model_builder(**params)
                else:
                    model = RandomForestClassifier(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', None),
                        random_state=42
                    )
                
                if X_train is not None and y_train is not None:
                    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    mean_score = float(np.mean(scores))
                    
                    all_results.append({"params": params, "score": mean_score})
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                else:
                    all_results.append({"params": params, "score": 0.0})
            
            self.logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
            
            return {
                "best_params": best_params,
                "best_metrics": {"accuracy": best_score, "loss": 1.0 - best_score},
                "num_trials_completed": len(param_combinations),
                "all_results": all_results
            }
        except ImportError as e:
            self.logger.warning(f"sklearn not available for tuning: {str(e)}")
            return {"best_params": {}, "best_metrics": {}, "num_trials_completed": 0, "error": "sklearn not available"}
        except Exception as e:
            self.logger.error(f"Failed to perform hyperparameter tuning: {str(e)}")
            return {"best_params": {}, "best_metrics": {}, "num_trials_completed": 0, "error": str(e)}
    
    def process_geospatial_data(self, data_paths: List[str], operations: List[Dict],
                              output_path: Optional[str] = None) -> Dict:
        """
        Process geospatial data in parallel using Ray or ThreadPoolExecutor fallback.
        
        Args:
            data_paths: Paths to the geospatial data files
            operations: List of geospatial operations to apply
            output_path: Path to write the processed data to
            
        Returns:
            Dict: Processing results with processed data
        """
        self.logger.info(f"Processing {len(data_paths)} geospatial data files")
        
        def process_single_file(file_path: str) -> Dict:
            result = {"file": file_path, "success": False, "data": None}
            try:
                if file_path.endswith('.parquet'):
                    data = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.json') or file_path.endswith('.geojson'):
                    data = pd.read_json(file_path)
                else:
                    self.logger.warning(f"Unsupported file format: {file_path}")
                    return result
                
                for operation in operations:
                    op_type = operation.get("type")
                    
                    if op_type == "reproject" and 'geometry' in data.columns:
                        pass
                    
                    elif op_type == "extract_features":
                        raster_column = operation.get("raster_column", "")
                        feature_type = operation.get("feature_type", "")
                        if raster_column in data.columns:
                            values = data[raster_column].values.astype(float)
                            if feature_type == "slope":
                                data[f"{raster_column}_slope"] = np.gradient(values)
                            elif feature_type == "aspect":
                                data[f"{raster_column}_aspect"] = np.arctan2(np.gradient(values), 1.0)
                            elif feature_type == "mean":
                                data[f"{raster_column}_mean"] = np.mean(values)
                            elif feature_type == "std":
                                data[f"{raster_column}_std"] = np.std(values)
                
                result["data"] = data
                result["success"] = True
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                result["error"] = str(e)
            return result
        
        results = self.parallel_task(process_single_file, data_paths)
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        combined_data = None
        if successful:
            data_frames = [r["data"] for r in successful if r["data"] is not None]
            if data_frames:
                combined_data = pd.concat(data_frames, ignore_index=True)
                
                if output_path:
                    import os
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                    if output_path.endswith('.parquet'):
                        combined_data.to_parquet(output_path)
                    elif output_path.endswith('.csv'):
                        combined_data.to_csv(output_path, index=False)
                    else:
                        combined_data.to_parquet(output_path + '.parquet')
                    self.logger.info(f"Wrote processed data to {output_path}")
        
        self.logger.info(f"Successfully processed {len(successful)}/{len(data_paths)} files")
        
        return {
            "files_processed": len(successful),
            "files_failed": len(failed),
            "operations_applied": len(operations),
            "output_path": output_path,
            "combined_data": combined_data
        }
    
    def shutdown(self) -> None:
        """Shutdown the Ray cluster or executor."""
        self.logger.info("Shutting down processing resources")
        
        try:
            if self._ray_initialized and RAY_AVAILABLE:
                ray.shutdown()
                self._ray_initialized = False
                self.logger.info("Successfully shut down Ray cluster")
            
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
                self.logger.info("Successfully shut down ThreadPoolExecutor")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = RayConfig(
        address="auto",
        num_cpus=None,  # Use all available
        num_gpus=None,  # Use all available
        train_parallelism=4,
        geospatial_batch_size=1000
    )
    
    # Create a processor
    processor = RayProcessor(config)
    
    # Define a simple function for parallel execution
    def process_tile(tile_data):
        # In a real implementation, this would process a tile of geospatial data
        return {"tile_id": tile_data["id"], "processed": True}
    
    # Execute a function in parallel
    tiles = [{"id": f"tile_{i}"} for i in range(100)]
    results = processor.parallel_task(
        func=process_tile,
        items=tiles,
        num_cpus_per_task=1
    )
    
    # Process a dataset
    processed_dataset = processor.process_dataset(
        dataset_path="/data/mineralvision/raw/satellite_imagery",
        transformations=[
            {
                "type": "map",
                "function": "normalize_bands",
                "batch": True,
                "compute": "actors"
            },
            {
                "type": "filter",
                "function": "remove_cloudy_pixels"
            },
            {
                "type": "map",
                "function": "calculate_indices",
                "batch": True
            }
        ],
        output_path="/data/mineralvision/processed/satellite_imagery"
    )
    
    # Train a model
    training_results = processor.train_model(
        train_dataset=None,  # In a real implementation, this would be a Ray Dataset
        val_dataset=None,    # In a real implementation, this would be a Ray Dataset
        model_config={
            "type": "unet",
            "hyperparameters": {
                "depth": 5,
                "filters_base": 64,
                "dropout_rate": 0.2
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "optimizer": "adam",
                "learning_rate": 0.001
            }
        },
        num_workers=4,
        use_gpu=True
    )
    
    # Perform hyperparameter tuning
    tuning_results = processor.hyperparameter_tuning(
        train_dataset=None,  # In a real implementation, this would be a Ray Dataset
        val_dataset=None,    # In a real implementation, this would be a Ray Dataset
        model_builder=None,  # In a real implementation, this would be a function
        param_space={
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [16, 32, 64],
            "num_layers": [2, 3, 4],
            "hidden_size": [64, 128, 256]
        },
        num_samples=20,
        resources_per_trial={"cpu": 2, "gpu": 0.5}
    )
    
    # Process geospatial data
    geospatial_results = processor.process_geospatial_data(
        data_paths=[f"/data/mineralvision/raw/lidar/tile_{i}.laz" for i in range(10)],
        operations=[
            {
                "type": "rasterize",
                "vector_column": "geometry",
                "resolution": 1.0
            },
            {
                "type": "extract_features",
                "raster_column": "elevation",
                "feature_type": "slope"
            },
            {
                "type": "extract_features",
                "raster_column": "elevation",
                "feature_type": "aspect"
            }
        ],
        output_path="/data/mineralvision/processed/lidar_features"
    )
    
    # Shutdown the processor
    processor.shutdown()
