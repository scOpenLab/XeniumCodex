# CODEX to Xenium Registration Tools

- XeniumCodexRegistrator.py
- RegisteredCoreSplitter.py

## XeniumCodexRegistrator.py

This script aligns a CODEX image with a Xenium spatial transcriptomics dataset and saves the result as a [`SpatialData`](https://spatialdata.scverse.org/) object, with the registered CODEX image and filtered Xenium transcript table.

### Usage

```bash
python3 XeniumCodexRegistrator.py \
  <xenium_folder> \
  <xenium_zarr> \
  <codex_zarr> \
  <codex_channels.txt> \
  <output_path>
```

| Argument             | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| `xenium_folder`      | Path to the Xenium experiment folder (from 10x Genomics)         |
| `xenium_zarr`        | Path to the Xenium morphology focus DAPI image (OME-Zarr format) |
| `codex_zarr`         | Path to the CODEX image (OME-Zarr format)                        |
| `codex_channels.txt` | Text file listing CODEX channel names, one per line              |
| `output_path`        | Output path where the `SpatialData` Zarr will be saved           |

### Workflow

1. **Image Loading**
   CODEX and Xenium DAPI images are loaded and processed using `dask.array`.

2. **Downscaling**
   Both images are downscaled to fit into memory for SIFT landmark detection (8X).

3. **Image Registration**

   * SIFT keypoints are computed with OpenCV.
   * Keypoint matches are filtered using Lowe’s ratio test (0.4).
   * An affine transformation is estimated from matched points and rescaled up (8X).

4. **SpatialData Integration**

   * The CODEX image is added to the Xenium `SpatialData` object.
   * An affine transformation aligning CODEX to Xenium is added.

5. **Transcript Table Cleanup**
The following probes/codewords are removed:
- "deprecated_codeword"
- "negative_control_probe"
- "genomic_control_probe"
- "negative_control_codeword"
- "unassigned_codeword"

The following columns are removed:
- "z"
- "fov_name"
- "nucleus_distance"
- "codeword_index"
- "codeword_category"
- "is_gene"
- "transcript_id"
- "overlaps_nucleus"

7. **Output**
   The resulting `SpatialData` object is saved to the specified `output_path`.

### Output

The output is a single `SpatialData` object, with associated Zarr store: 
```
├── Images
│     ├── 'codex': DataTree[cyx] (multiple resolutions)
│     └── 'morphology_focus': DataTree[cyx] (multiple resolutions)
├── Labels
│     └── 'cell_labels': DataTree[yx] (multiple resolutions)
├── Points
│     └── 'transcripts': DataFrame with shape: (<Delayed>, 5) (2D points)
├── Shapes
│     └── 'cell_boundaries': GeoDataFrame shape: (N cells, 1) (2D shapes)
└── Tables
      └── 'table': AnnData (N cells, N features)
with coordinate systems:
    ▸ 'codex', with elements:
        codex (Images)
    ▸ 'global', with elements:
        codex (Images), morphology_focus (Images), cell_labels (Labels), transcripts (Points), cell_boundaries (Shapes)
```

The object contains two coordinate spaces:
- "codex": original codex coordinates (you can ignore this)
- "global": Xenium + aligned codex data (this should be used for visualization)


## RegisteredCoreSplitter.py

This script crops an aligned Xenium `SpatialData` object generated with the previous script,
into multiple subregions based on bounding boxes provided in a CSV file. Each cropped region is written as an individual `.zarr` dataset using the same structure as the original input.

### Usage

```bash
python3 RegisteredCoreSplitter.py \
<input_folder> \
<bounding_boxes> \
<output_folder>
```

| Argument            | Description |
|---------------------|-------------|
| `input_folder`      | Path to the input `SpatialData` Zarr folder |
| `bounding_boxes`    | Path to a CSV file with bounding box definitions |
| `output_folder`     | Path to the output folder where cropped `.zarr` files will be written |

The CSV file should have five columns with the following format:

```
core_id,x_min,x_max,y_min,y_max
1,5000,15000,7000,17000
2,16000,26000,9000,19000
```

- Pixel coordinates are assumed to be in **micrometers** and to apply to the Xenium morphology focus DAPI image and will be converted to pixel units (1 px = 212 nm).
- Each row defines one rectangular crop region.

### Workflow

1. **Load Data**  
   Read the full `SpatialData` object from the specified `input_folder`.

2. **Read Bounding Boxes**  
   Load the CSV file and convert coordinates from micrometers to pixels.

3. **Crop and Save**  
   For each bounding box:
   - Crop the `SpatialData` using `bounding_box_query`.
   - Rechunk the image and label data for efficient storage.
   - Save the cropped region to a `.zarr` file named `core-<core_id>.zarr` in the `output_folder`.

### Output

Each cropped dataset will be saved in the `output_folder` as:

```
core-<core_id>.zarr
```

These outputs retain the same structure as the original `SpatialData` object.
