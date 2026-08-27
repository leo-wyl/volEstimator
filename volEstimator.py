#!/usr/bin/env python3
"""Align a PLY to its plate, split it, and allocate mesh-area weight.

The script uses every point without colour filtering, erosion, or cropping. It
finds the dominant plate plane, rotates its normal onto Z, and places that plane
at Z=0 before clustering and rendering. A top-down 2.5D Delaunay mesh gives
each region an area-weighted height score instead of a point-density score.
Input distances must already be centimetres for weight prediction. Part
weights sum to the whole prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import Delaunay, QhullError


MODEL_INTERCEPT_G = 50.15553214510801
MODEL_SLOPE_G_PER_CM = 1.864809347201741
MESH_CELLS_ALONG_LONG_AXIS = 250
MESH_MAX_EDGE_CELLS = 2.5
MESH_HISTOGRAM_BINS = 30


def align_ground_plane_to_xy(
    points: np.ndarray,
    *,
    margin_fraction: float = 0.20,
    distance_threshold: float | None = None,
    ransac_iterations: int = 600,
    random_seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Estimate the ground from the footprint margin and map it onto Z=0.

    The outer footprint ring is used for RANSAC, preventing a dense central
    plant or object from being selected as the ground. Returned coordinates
    satisfy ``aligned = (points - origin) @ basis``; the third basis column is
    the fitted ground normal.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("Expected at least eight XYZ points, got {}".format(points.shape))
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates.")
    if not 0.05 <= margin_fraction < 0.50:
        raise ValueError("margin_fraction must be in [0.05, 0.50).")
    if ransac_iterations < 1:
        raise ValueError("ransac_iterations must be positive.")

    robust_center = np.median(points, axis=0)
    _, _, preliminary_axes = np.linalg.svd(
        points - robust_center, full_matrices=False
    )
    footprint = (points - robust_center) @ preliminary_axes[:2].T
    footprint_low, footprint_high = np.percentile(footprint, [1.0, 99.0], axis=0)
    half_span = np.maximum((footprint_high - footprint_low) / 2.0, 1e-12)
    footprint_midpoint = (footprint_low + footprint_high) / 2.0
    normalized_footprint = np.abs((footprint - footprint_midpoint) / half_span)
    margin_mask = np.any(
        normalized_footprint >= 1.0 - margin_fraction, axis=1
    )
    margin_points = points[margin_mask]
    if len(margin_points) < 8:
        raise ValueError("The footprint margin contains too few points.")

    scene_scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    plane_threshold = (
        max(0.005 * scene_scale, 1e-6)
        if distance_threshold is None
        else float(distance_threshold)
    )
    if not np.isfinite(plane_threshold) or plane_threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive.")

    rng = np.random.default_rng(random_seed)
    sample = margin_points
    if len(sample) > 20_000:
        sample = sample[rng.choice(len(sample), 20_000, replace=False)]
    best_count = 0
    best_origin = None
    best_normal = None
    for _ in range(ransac_iterations):
        triangle = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            continue
        normal /= length
        distances = np.abs((sample - triangle[0]) @ normal)
        count = int(np.count_nonzero(distances <= plane_threshold))
        if count > best_count:
            best_count = count
            best_origin = triangle[0]
            best_normal = normal
    if best_origin is None or best_count < max(8, int(0.08 * len(sample))):
        raise ValueError("Could not identify a stable ground plane in the margin.")

    ground_origin = best_origin
    ground_normal = best_normal
    for _ in range(3):
        in_margin_plane = (
            np.abs((margin_points - ground_origin) @ ground_normal)
            <= plane_threshold
        )
        plane_points = margin_points[in_margin_plane]
        if len(plane_points) < 3:
            raise ValueError("Ground-plane refinement retained too few points.")
        ground_origin = plane_points.mean(axis=0)
        _, _, vectors = np.linalg.svd(plane_points - ground_origin, full_matrices=False)
        refined_normal = vectors[-1]
        if float(np.dot(refined_normal, ground_normal)) < 0:
            refined_normal *= -1.0
        ground_normal = refined_normal

    signed_height = (points - ground_origin) @ ground_normal
    if float(np.percentile(signed_height, 98.0)) < -float(
        np.percentile(signed_height, 2.0)
    ):
        ground_normal *= -1.0
        signed_height *= -1.0
    all_ground_mask = np.abs(signed_height) <= plane_threshold

    plane_centered = plane_points - ground_origin
    plane_projected = plane_centered - np.outer(
        plane_centered @ ground_normal, ground_normal
    )
    _, _, plane_axes = np.linalg.svd(plane_projected, full_matrices=False)
    x_axis = plane_axes[0]
    x_axis -= float(np.dot(x_axis, ground_normal)) * ground_normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(ground_normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack((x_axis, y_axis, ground_normal))
    aligned = (points - ground_origin) @ basis
    ground_residuals = aligned[margin_mask & all_ground_mask, 2]

    metadata = {
        "method": "footprint_margin_ransac_ground_to_xy",
        "ground_margin_fraction": margin_fraction,
        "ground_margin_points": int(len(margin_points)),
        "ground_plane_margin_inliers": int(len(plane_points)),
        "ground_plane_all_inliers": int(np.count_nonzero(all_ground_mask)),
        "ground_plane_threshold_input_units": plane_threshold,
        "ground_plane_rmse_input_units": float(
            np.sqrt(np.mean(np.square(ground_residuals)))
        ),
        "ground_origin_input_coordinates": ground_origin.tolist(),
        "ground_normal_input_coordinates": ground_normal.tolist(),
        "input_to_aligned_basis": basis.tolist(),
        "aligned_span_xyz_input_units": np.ptp(aligned, axis=0).tolist(),
    }
    return aligned, metadata


def split_point_cloud(ply_path: Path) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], dict]:
    """Load, plate-align, visualize, and split every point into two XY clusters."""
    cloud = trimesh.load(ply_path, process=False)
    if isinstance(cloud, trimesh.Scene):
        if not cloud.geometry:
            raise ValueError("PLY contains no geometry: {}".format(ply_path))
        cloud = cloud.dump(concatenate=True)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("Expected at least eight XYZ points, got {}".format(points.shape))
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates.")

    points, alignment = align_ground_plane_to_xy(points)
    plane_threshold = alignment["ground_plane_threshold_input_units"]

    split_fit_mask = points[:, 2] > plane_threshold
    if np.count_nonzero(split_fit_mask) < 8:
        split_fit_mask = np.ones(len(points), dtype=bool)
    split_fit_xy = points[split_fit_mask, :2]
    xy_center = np.median(split_fit_xy, axis=0)
    covariance = np.cov((split_fit_xy - xy_center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    split_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    fit_coordinate = (split_fit_xy - xy_center) @ split_axis

    centers = np.percentile(fit_coordinate, [25.0, 75.0]).astype(np.float64)
    for _ in range(100):
        first_fit = np.abs(fit_coordinate - centers[0]) <= np.abs(
            fit_coordinate - centers[1]
        )
        if first_fit.all() or (~first_fit).all():
            raise ValueError("Could not form two non-empty point clusters.")
        updated = np.array(
            [fit_coordinate[first_fit].mean(), fit_coordinate[~first_fit].mean()],
            dtype=np.float64,
        )
        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break
        centers = updated

    if centers[0] > centers[1]:
        centers = centers[::-1]
        split_axis *= -1.0
    split_threshold = float(centers.mean())
    coordinate = (points[:, :2] - xy_center) @ split_axis
    first = coordinate <= split_threshold
    parts = (points[first], points[~first])
    if sum(len(part) for part in parts) != len(points):
        raise RuntimeError("The split did not preserve every input point.")

    image_paths = []
    rng = np.random.default_rng(42)
    for part_number, part in enumerate(parts, start=1):
        displayed = part
        if len(displayed) > 100_000:
            indices = rng.choice(len(displayed), 100_000, replace=False)
            displayed = displayed[indices]
        figure, axis = plt.subplots(figsize=(8, 7))
        plot = axis.scatter(
            displayed[:, 0],
            displayed[:, 1],
            c=displayed[:, 2],
            cmap="viridis",
            s=0.35,
            alpha=0.75,
            rasterized=True,
        )
        axis.set(
            xlabel="Aligned X (input units)",
            ylabel="Aligned Y (input units)",
            title="{} - part {} top-down".format(ply_path.stem, part_number),
        )
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(plot, ax=axis, label="Height above plate (input units)")
        figure.tight_layout()
        image_path = ply_path.with_name(
            "{}_part{}_topdown.png".format(ply_path.stem, part_number)
        )
        figure.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        image_paths.append(str(image_path.resolve()))

    metadata = {
        **alignment,
        "method": "ground_margin_alignment_then_above_ground_xy_two_means",
        "plate_plane_inliers": alignment["ground_plane_all_inliers"],
        "plate_plane_threshold_input_units": plane_threshold,
        "plate_normal_input_coordinates": alignment[
            "ground_normal_input_coordinates"
        ],
        "axis_xy": split_axis.tolist(),
        "threshold_on_axis_input_units": split_threshold,
        "threshold_on_axis_cm": split_threshold,
        "split_fit_above_ground_points": int(np.count_nonzero(split_fit_mask)),
        "input_points": int(len(points)),
        "part_points": [int(len(part)) for part in parts],
        "topdown_images": image_paths,
    }
    return points, parts, metadata


def calculate_mesh_area_weights(
    points: np.ndarray, parts: tuple[np.ndarray, np.ndarray], ply_path: Path
) -> dict:
    """Allocate P95-model weight using a density-independent 2.5D mesh."""
    xy_min = points[:, :2].min(axis=0)
    long_axis = float(np.ptp(points[:, :2], axis=0).max())
    cell_size = max(long_axis / MESH_CELLS_ALONG_LONG_AXIS, 1e-9)
    max_edge = MESH_MAX_EDGE_CELLS * cell_size
    height_max = max(float(points[:, 2].max()), cell_size)
    height_edges = np.linspace(0.0, height_max, MESH_HISTOGRAM_BINS + 1)

    mesh_results = []
    for part_number, part in enumerate(parts, start=1):
        cell_indices = np.floor((part[:, :2] - xy_min) / cell_size).astype(np.int64)
        unique_cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
        canopy_height = np.zeros(len(unique_cells), dtype=np.float64)
        np.maximum.at(canopy_height, inverse, np.maximum(part[:, 2], 0.0))
        mesh_xy = xy_min + (unique_cells.astype(np.float64) + 0.5) * cell_size
        if len(mesh_xy) < 3:
            raise ValueError("A split part has fewer than three occupied XY mesh cells.")

        try:
            triangles = Delaunay(mesh_xy, qhull_options="QJ").simplices
        except QhullError as error:
            raise ValueError("Could not build the top-down Delaunay mesh.") from error

        triangle_xy = mesh_xy[triangles]
        edge_01 = np.linalg.norm(triangle_xy[:, 0] - triangle_xy[:, 1], axis=1)
        edge_12 = np.linalg.norm(triangle_xy[:, 1] - triangle_xy[:, 2], axis=1)
        edge_20 = np.linalg.norm(triangle_xy[:, 2] - triangle_xy[:, 0], axis=1)
        local_triangles = np.maximum.reduce((edge_01, edge_12, edge_20)) <= max_edge
        triangles = triangles[local_triangles]
        triangle_xy = triangle_xy[local_triangles]
        if len(triangles) == 0:
            raise ValueError("The mesh has no local triangles after rejecting gap bridges.")

        twice_area = np.abs(
            (triangle_xy[:, 1, 0] - triangle_xy[:, 0, 0])
            * (triangle_xy[:, 2, 1] - triangle_xy[:, 0, 1])
            - (triangle_xy[:, 1, 1] - triangle_xy[:, 0, 1])
            * (triangle_xy[:, 2, 0] - triangle_xy[:, 0, 0])
        )
        projected_area = 0.5 * twice_area
        triangle_height = canopy_height[triangles].mean(axis=1)
        area_height = float(projected_area @ triangle_height)
        area_by_height, _ = np.histogram(
            triangle_height, bins=height_edges, weights=projected_area
        )
        height_centers = (height_edges[:-1] + height_edges[1:]) / 2.0
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            height_centers,
            area_by_height,
            width=np.diff(height_edges),
            align="center",
            color="#2a9d8f",
            edgecolor="#174f49",
            linewidth=0.35,
        )
        axis.set(
            xlabel="Triangle mean height above plate (input units)",
            ylabel="Projected mesh area (input units²)",
            title="{} - part {} area-weighted mesh height histogram".format(
                ply_path.stem, part_number
            ),
            xlim=(0.0, height_max),
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        histogram_path = ply_path.with_name(
            "{}_part{}_mesh_height_hist.png".format(ply_path.stem, part_number)
        )
        figure.savefig(histogram_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        mesh_results.append(
            {
                "mesh_cells": int(len(mesh_xy)),
                "mesh_triangles": int(len(triangles)),
                "projected_area_square_input_units": float(projected_area.sum()),
                "area_weighted_height_cubic_input_units": area_height,
                "estimated_volume_cubic_input_units": area_height,
                "mesh_histogram": str(histogram_path.resolve()),
            }
        )

    mesh_total = sum(item["area_weighted_height_cubic_input_units"] for item in mesh_results)
    if mesh_total <= 0:
        raise ValueError("The meshes have no positive area-weighted height.")

    whole_p95_height = float(np.percentile(points[:, 2], 95.0))
    predicted_total = MODEL_INTERCEPT_G + MODEL_SLOPE_G_PER_CM * whole_p95_height
    shares = tuple(
        item["area_weighted_height_cubic_input_units"] / mesh_total
        for item in mesh_results
    )
    weights = tuple(predicted_total * share for share in shares)
    for part_number, (item, share, weight) in enumerate(
        zip(mesh_results, shares, weights), start=1
    ):
        item.update(
            {
                "part": part_number,
                "mesh_share": share,
                "predicted_weight_g": weight,
            }
        )
    return {
        "method": "area_weighted_2_5d_delaunay_mesh",
        "mesh_cell_size_input_units": cell_size,
        "mesh_max_edge_input_units": max_edge,
        "mesh_histogram_bins": MESH_HISTOGRAM_BINS,
        "estimated_total_volume_cubic_input_units": mesh_total,
        "whole_p95_height_input_units": whole_p95_height,
        "predicted_total_weight_g": predicted_total,
        "parts": mesh_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.ply.is_file():
        parser.error("PLY does not exist: {}".format(arguments.ply))

    all_points, two_parts, split_info = split_point_cloud(arguments.ply)
    result = calculate_mesh_area_weights(all_points, two_parts, arguments.ply)

    print("Split method: {}".format(split_info["method"]))
    print(
        "Ground margin: {} points; plane inliers: {} margin / {} total".format(
            split_info["ground_margin_points"],
            split_info["ground_plane_margin_inliers"],
            split_info["ground_plane_all_inliers"],
        )
    )
    print(
        "Aligned XYZ span: {:.3f} x {:.3f} x {:.3f} input units".format(
            *split_info["aligned_span_xyz_input_units"]
        )
    )
    print(
        "Point counts: {} + {} = {} (split fitted with {} above-ground points)".format(
            split_info["part_points"][0],
            split_info["part_points"][1],
            split_info["input_points"],
            split_info["split_fit_above_ground_points"],
        )
    )
    print("Volume method: {}".format(result["method"]))
    print(
        "Mesh cell size: {:.6f}, maximum local edge: {:.6f} input units".format(
            result["mesh_cell_size_input_units"], result["mesh_max_edge_input_units"]
        )
    )
    print(
        "Estimated total volume: {:.6g} input units^3".format(
            result["estimated_total_volume_cubic_input_units"]
        )
    )
    print(
        "Whole P95 height: {:.3f} input units (treated as cm by the fitted model)".format(
            result["whole_p95_height_input_units"]
        )
    )
    print("Predicted total weight: {:.3f} g".format(result["predicted_total_weight_g"]))
    print("Part 1 top-down image: {}".format(split_info["topdown_images"][0]))
    print("Part 2 top-down image: {}".format(split_info["topdown_images"][1]))
    for part in result["parts"]:
        print(
            "Part {}: cells={}, triangles={}, area={:.6g} unit^2, volume={:.6g} unit^3, share={:.3%}, predicted weight={:.3f} g".format(
                part["part"],
                part["mesh_cells"],
                part["mesh_triangles"],
                part["projected_area_square_input_units"],
                part["estimated_volume_cubic_input_units"],
                part["mesh_share"],
                part["predicted_weight_g"],
            )
        )
        print("Part {} mesh histogram: {}".format(part["part"], part["mesh_histogram"]))

if __name__ == "__main__":
    main()
