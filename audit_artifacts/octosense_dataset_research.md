# OctoSense Dataset Research Notes

**Dataset page:** https://huggingface.co/datasets/Voxel51/OctoSense  
**API metadata:** https://huggingface.co/api/datasets/Voxel51/OctoSense  
**File listing:** https://huggingface.co/api/datasets/Voxel51/OctoSense/tree/main  
**Sample manifest:** https://huggingface.co/datasets/Voxel51/OctoSense/raw/main/samples.json  
**Metadata manifest:** https://huggingface.co/datasets/Voxel51/OctoSense/raw/main/metadata.json

## Verified Source Facts

The Hugging Face dataset API identifies `Voxel51/OctoSense` as public and ungated, tagged `license:mit`, `library:fiftyone`, `multimodal`, `mcap`, and `robotics`. The dataset card describes it as a time-synchronized, calibrated multi-sensor robot-perception dataset from an open-source eight-sensor rig collected across three platforms: a car, a boat, and a Unitree Go2-W quadruped. It packages eight representative episodes as MCAP files, one sample per episode, intended for synchronized playback in the FiftyOne multimodal viewer.

The repository includes `data/`, `README.md`, `fiftyone.yml`, `metadata.json`, `samples.json`, and a preview GIF. The user-facing Hugging Face viewer currently fails with a `StreamingRowsError` / `CastError` because the FiftyOne JSON structure does not match the viewer’s inferred columns. This is a source usability issue, not evidence that the MCAP assets themselves are invalid.

The metadata manifests define a multimodal sample schema including: MCAP filepath, duration, message and channel count, topic and schema lists, session, start time, split, LiDAR/RGB/IMU/GPS counts, GPS quality and bounds, calibration identifiers, event-camera counts, platform, and optional operational properties. The samples manifest confirms available topic families include infrared, stereo/left/right RGB, left/right event cameras, LiDAR point clouds, IMUs, GPS, calibration, TF transforms, and platform-specific topics. Car episodes additionally expose odometry, captions, depth, flow, semantic topics, radar, and vehicle controls. Sample metadata includes `has_seg` for car episodes, but the source does not establish an oil-spill, mineral, drilling, or environmental ground-truth task.

## Initial MineralVision Fit

Best fit is governed **multimodal sensor-ingestion and data-quality evaluation**: calibration/clock alignment checks, sensor-dropout and GPS-quality policy testing, LiDAR/RGB/IMU/GPS ingestion contracts, evidence provenance, and human-review UX. It is not a domain-valid training or accuracy-validation source for mineral prospectivity or oil-spill segmentation. It should never be combined with oil-spill holdouts or used to substantiate mineral or spill model accuracy.

## License and Governance Note

The source API reports an MIT license. The implementation must capture the source repository, revision SHA `0ff6263bdaf0693ea039016398624501c4812804`, file digest/size, license assertion, intended use, lineage hash, and an explicit domain-fit declaration before data is admitted. The external source and its terms should be re-checked before downloading or redistributing assets.

## Corroborating Research

The OctoSense paper abstract (arXiv:2606.27317) describes a 59-hour time-synchronized robotics collection with stereo RGB/event cameras, LiDAR, thermal imagery, IMU, RTK-corrected GPS, and platform proprioception. It positions the material for multimodal self-supervised robotics work under heterogeneous sampling rates, latencies, noise, night-time, and degraded-sensor conditions. That corroborates a controlled use for generic sensor fusion, representation learning, calibration, synchronization, and modality-degradation evaluation. It does not add a mineral, oil-spill, environmental incident, or geological target label.

## Implementation Decision

Implement a local, manifest-first adapter under `src/api/innovations/octosense`. The adapter will not download remote content, execute dataset code, or expose an unauthenticated upload/download API. It will take reviewer-supplied local `samples.json` and `metadata.json`, validate them against a deliberately narrow schema, normalize the episode inventory, calculate content digests and deterministic lineage, create a dataset quality report, and emit an evidence-payload contract compatible with the existing tenant-bound governed integration hub. A CLI will accept only approved purposes; it will fail closed for oil-spill/mineral evaluation and other unsupported claims. Small checked-in fixture manifests will test the adapter without downloading raw MCAP assets.

**Paper source:** https://arxiv.org/abs/2606.27317
