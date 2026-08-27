# Volume Estimator

`volEstimator.py` estimates the fresh weight of two plant subunits from a PLY
point cloud. It aligns the cloud to the tray or plate, divides all points into
two spatial clusters, constructs a top-down 2.5D mesh for each cluster, and
allocates a whole-tray weight prediction between the two parts.

> [!IMPORTANT]
> Input coordinates must be expressed in **centimetres**. The fitted prediction
> model treats the point-cloud height as centimetres; using metres or another
> unit will produce an invalid weight estimate.

## How it works

1. Finds points around the outer footprint of the cloud.
2. Fits the dominant plate plane with RANSAC and aligns it to `Z=0`.
3. Splits the above-ground points into two clusters along their principal XY
   axis, then assigns every input point to one of those parts.
4. Builds a Delaunay mesh over the occupied XY cells in each part, rejecting
   triangles that bridge large gaps.
5. Uses projected triangle area and canopy height to calculate each part's
   share of the total.
6. Predicts whole-tray fresh weight from the 95th-percentile height and divides
   that prediction according to the mesh shares.

The script uses every point; it does not filter by colour, erode the cloud, or
crop it.

## Requirements

- Python 3.9 or later
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [trimesh](https://trimesh.org/)

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib trimesh
```

## Usage

Run the estimator with a PLY file:

```bash
python volEstimator.py --ply mark1mark2Seg.ply
```

The command prints alignment diagnostics, point counts, mesh statistics,
estimated total volume, total predicted weight, and the weight assigned to
each part. Use `--help` to display the command-line options:

```bash
python volEstimator.py --help
```

### Generated files

Images are written beside the input PLY, using its filename stem:

- `<stem>_part1_topdown.png`
- `<stem>_part2_topdown.png`
- `<stem>_part1_mesh_height_hist.png`
- `<stem>_part2_mesh_height_hist.png`

The top-down plots show aligned points coloured by height. The histograms show
projected mesh area grouped by triangle mean height. Existing files with these
names are overwritten.

## Repository data

The repository includes example segmented and unsegmented PLY point clouds:

- `0707leafPointSeg.ply` and `0707leafPointNoSeg.ply`
- `mark1mark2Seg.ply` and `mark1mark2NoSeg.ply`

`current_dataset_fresh_weights.csv` contains fresh-weight measurements and
records which observations were included in regression development. The
runtime script does **not** read this CSV: the current regression intercept and
slope are constants in `volEstimator.py`.

## Assumptions and limitations

- The cloud must contain at least eight finite XYZ points and enough outer
  plate points to fit a stable ground plane.
- The scene is expected to have two spatially separable subunits on a visible,
  approximately planar plate or tray.
- Each resulting part must occupy enough XY cells to construct a local mesh.
- Weight predictions are specific to the fitted model and its source data;
  validate the model before applying it to new crops, imaging systems, or
  acquisition conditions.
- Generated volume values are geometric proxies based on a 2.5D canopy mesh,
  not watertight physical volume measurements.

## Project structure

```text
volEstimator.py                    # Command-line estimator
current_dataset_fresh_weights.csv # Fresh-weight/model-development data
*.ply                              # Example point clouds
```
