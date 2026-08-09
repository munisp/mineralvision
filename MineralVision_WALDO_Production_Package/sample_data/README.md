# Sample Data for MineralVision WALDO Integration

This directory contains sample data for testing and demonstrating the MineralVision WALDO integration. The data is organized into three categories:

1. Aerial Imagery: High-resolution aerial images with various objects for detection
2. Drone Video: Video footage from drones with moving objects
3. Historical Data: Pre-processed detection results for analysis and visualization

## Data Usage Guidelines

- This sample data is provided for demonstration and testing purposes only
- The data may be used for training and evaluation of the WALDO integration
- Do not use this data for production deployments without proper validation
- For production use, replace with your own data specific to your mining operations

## Data Sources

The sample data is derived from the following sources:

- Aerial imagery: Synthetic data generated for mining operations
- Drone video: Captured at test mining sites with permission
- Historical data: Generated from previous analyses of mining operations

## Data Format

### Aerial Imagery

- Format: JPEG/PNG
- Resolution: 4K (3840x2160)
- Metadata: EXIF data including GPS coordinates, altitude, and camera parameters
- Annotations: JSON files with bounding box coordinates and class labels

### Drone Video

- Format: MP4 (H.264)
- Resolution: 4K (3840x2160)
- Frame Rate: 30 fps
- Duration: 2-5 minutes per clip
- Metadata: JSON files with flight path, altitude, and camera parameters

### Historical Data

- Format: JSON
- Structure: Detection results including bounding boxes, class labels, confidence scores, and timestamps
- Includes: Tracking information, measurement data, and geospatial coordinates

## License

This sample data is provided under the MineralVision Sample Data License, which allows for:

- Use for evaluation, testing, and demonstration
- Modification for integration testing
- Distribution as part of the MineralVision WALDO package

For questions regarding the use of this data, please contact data@mineralvision.com.
