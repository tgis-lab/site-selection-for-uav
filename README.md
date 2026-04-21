# Drone Nest Site Selection and Algorithm Comparison

This repository provides a workflow for **drone nest site selection in mountainous environments** using an enhanced Particle Swarm Optimization method and several baseline optimization algorithms. It also includes utilities for **algorithm comparison**, **result export**, and **2D/3D visualization**.

The project is designed for scenarios such as emergency logistics, mountainous deployment planning, and digital-twin-based infrastructure optimization.

## Overview

The repository contains four main parts:

- `V9EAPSO.py`: the main enhanced adaptive PSO implementation for drone nest siting.
- `algorithm_comparison.py`: the core comparison engine for running multiple optimization methods on the same dataset.
- `run_algorithm_comparison.py`: a command-line wrapper for launching comparison experiments.
- `run_visualization.py` and `visualization_tool.py`: utilities for generating 2D and 3D visual outputs from saved results.

The comparison workflow evaluates several algorithms under the same terrain, building, and road constraints:

- K-means
- GA
- NSGA-II
- ACO
- PSO / EAPSO

## Main Features

- Terrain-aware drone nest site selection based on DEM data
- Building-constrained candidate generation for feasible nest placement
- Road-based task-point generation
- Line-of-sight-aware fitness evaluation
- Energy-aware deployment penalty modeling
- Adaptive PSO with dynamic parameter update and diversity preservation
- Multi-algorithm benchmarking under a unified evaluation pipeline
- 2D comparison plots and 3D scene visualization
- Export of best solution coordinates in UTM, geographic, CSV, and SHP formats

## Recommended Repository Structure

The examples below assume the main scripts are placed under `algorithm_source_code/`.

```text
DRONES/
├─ README.md
├─ algorithm_source_code/
│  ├─ V9EAPSO.py
│  ├─ algorithm_comparison.py
│  ├─ run_algorithm_comparison.py
│  ├─ run_visualization.py
│  ├─ visualization_tool.py
│  └─ ...
└─ visualization_outputs/
```

If your GitHub version uses a slightly different folder layout, adjust the script paths accordingly.

## Requirements

This project is written in Python and depends on scientific computing, GIS, optimization, and visualization libraries.

Typical dependencies include:

```bash
pip install numpy pandas matplotlib plotly tqdm scipy scikit-learn rasterio geopandas shapely pyproj deap pymoo
```

Depending on your environment, `geopandas`, `rasterio`, and `pyogrio/fiona` may require a properly configured GIS stack.

## Input Data

The workflow expects three main input datasets:

- A DEM/terrain raster file in `.tif` format
- A building layer in `.shp` format
- A road network layer in `.shp` format

These datasets are used as follows:

- **Terrain data** provides elevation and slope information.
- **Building data** provides candidate nest locations.
- **Road data** is used to generate demand or task points.

## Core Script Roles

### `V9EAPSO.py`

This is the main optimization script for the enhanced adaptive PSO method. It is responsible for:

- loading terrain, building, and road data
- generating candidate sites and task points
- optimizing drone nest locations
- evaluating coverage, energy, and line-of-sight constraints
- exporting final solution files

Typical outputs include:

- `best_solution.csv`
- `geo_coordinates.csv`
- `nest_geodetic_coordinates.csv`
- `deployment_points.shp`
- log output in the console or log file

### `algorithm_comparison.py`

This module runs multiple algorithms on the same dataset and compares them using:

- fitness score
- coverage ratio
- execution time

It also supports:

- saving results to JSON
- generating comparison plots
- generating 3D visualizations

### `run_algorithm_comparison.py`

This is the recommended command-line entry point for algorithm benchmarking. It parses input arguments, launches the comparison routine, and writes summary outputs.

### `run_visualization.py`

This utility supports two modes:

- `run`: execute the comparison pipeline and save results
- `visualize`: reload a saved JSON file and generate figures again

### `visualization_tool.py`

This module reads saved algorithm results and produces:

- 2D comparative performance plots
- 3D visualization of terrain, task points, and deployment solutions

## Quick Start

Open a terminal and move to the script directory:

```bash
cd algorithm_source_code
```

## Run the Main EAPSO Optimizer

Run the standalone enhanced PSO workflow:

```bash
python V9EAPSO.py
```

This is the best choice if you want to:

- test only the improved PSO method
- export the final nest coordinates
- inspect the final deployment result for one scenario

## Run Multi-Algorithm Comparison

To compare multiple algorithms on the same dataset:

```bash
python run_algorithm_comparison.py ^
  --tiff "path\to\terrain.tif" ^
  --buildings "path\to\buildings.shp" ^
  --roads "path\to\roads.shp" ^
  --algorithms K-means GA NSGA-II ACO PSO ^
  --output "..\visualization_outputs\algorithm_comparison_results.png" ^
  --html "..\visualization_outputs\algorithm_comparison_3d.html"
```

If you want to run all supported algorithms, use:

```bash
python run_algorithm_comparison.py ^
  --tiff "path\to\terrain.tif" ^
  --buildings "path\to\buildings.shp" ^
  --roads "path\to\roads.shp" ^
  --algorithms all
```

## Run the Integrated Workflow

The integrated visualization utility supports both execution and re-visualization.

### Mode 1: Run and Save Results

```bash
python run_visualization.py run ^
  --tiff "path\to\terrain.tif" ^
  --buildings "path\to\buildings.shp" ^
  --roads "path\to\roads.shp" ^
  --algorithms K-means GA NSGA-II ACO PSO ^
  --output "..\visualization_outputs\algorithm_comparison_results.png" ^
  --html "..\visualization_outputs\algorithm_comparison_3d.html" ^
  --results "algorithm_results.json"
```

### Mode 2: Visualize Saved Results

```bash
python run_visualization.py visualize ^
  --results "algorithm_results.json" ^
  --output "..\visualization_outputs\visualization_results.png" ^
  --html "..\visualization_outputs\visualization_3d.html"
```

This mode is useful when you already have a saved JSON result file and only want to regenerate plots.

## Output Files

Depending on the script and mode, the project may generate:

- PNG comparison charts
- HTML 3D interactive visualization files
- JSON result files
- CSV coordinate exports
- SHP deployment point files
- console logs or log files

Typical visualization outputs include:

- `algorithm_comparison_results.png`
- `algorithm_comparison_3d.html`
- `visualization_results.png`
- `visualization_3d.html`

## Evaluation Logic

The fitness design generally combines three aspects:

- **coverage performance**
- **energy feasibility**
- **line-of-sight connectivity**

This means a solution with high coverage may still receive a lower fitness score if it violates communication visibility or energy-related constraints.

## Notes

- Run all commands from the directory that contains the target script, or update import paths accordingly.
- Some scripts in different local versions may reference slightly different module names. If needed, align the import statement with the module file present in your repository.
- GIS input files should use a consistent coordinate reference system.
- Large terrain and shapefile datasets may significantly increase runtime and memory usage.

## Suggested Usage Flow

For most users, the recommended workflow is:

1. Prepare the `.tif` terrain file and the `.shp` building and road layers.
2. Run `run_algorithm_comparison.py` or `run_visualization.py run`.
3. Inspect the generated PNG and HTML outputs.
4. If needed, rerun `run_visualization.py visualize` using the saved JSON file.
5. Run `V9EAPSO.py` separately when you want the final deployment solution export for the enhanced PSO method.

## License

Add your preferred license information here if you plan to publish the repository publicly.
