"""
Module providing registration between Xenium and CODEX data
Assumes that a morphology focus image and the CODEX image have been converted to zarr
The registration is done in two steps:
1) Coarse registration on the whole image
2) Finer registration by core
One SpatialData object is save d for each core
"""

import tempfile
import argparse
import dask
import dask_image
import skimage
import numpy as np
import cv2 as cv
import pandas as pd
import spatialdata as sd
import spatialdata_io
import tifffile
import zarr
from spatialdata.models import Image2DModel


def preprocess_reg(img, scale_factor):
    """
    Prepares an image for use with cv SIFT
    """
    matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    output_shape = np.asarray(img.shape)
    output_shape = (output_shape / scale_factor).astype("uint16")
    img = dask_image.ndinterp.affine_transform(
        img,
        matrix=matrix,
        output_shape=output_shape,
    )
    img = skimage.exposure.equalize_adapthist(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype("uint8")
    return img


def get_align_matrix(img1, img2, scale_factor=1):
    """
    Finds the matrix for the affine transform that aligns img1 to img2 (downscaled)
    the matrix is then: rescaled to the original size, and converted to work with
    a row/column convention on cyx images
    """

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = {"algorithm": 1, "trees": 5}  #  FLANN_INDEX_KDTREE = 1
    search_params = {"checks": 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    pt1 = []
    pt2 = []
    for n, m in enumerate(good):
        pt1.append(kp1[m.queryIdx].pt)
        pt2.append(kp2[m.trainIdx].pt)
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)

    landmarks1 = sd.models.ShapesModel.parse(pt1, geometry=0, radius=5)
    landmarks2 = sd.models.ShapesModel.parse(pt2, geometry=0, radius=5)

    affine = sd.transformations.get_transformation_between_landmarks(
        landmarks2, landmarks1
    )
    affine = affine.inverse().matrix
    affine[0:2, 2] *= scale_factor

    # Need to change matrix to adapt for xy vs row cols conventions
    # https://github.com/jni/affinder/issues/61
    def matrix_rc2xy(affine_matrix):
        swapped_cols = affine_matrix[:, [1, 0, 2]]
        swapped_rows = swapped_cols[[1, 0, 2], :]
        return swapped_rows

    affine = matrix_rc2xy(affine)

    # Make the matrix work with cyx images
    affine_cyx = np.eye(4)
    affine_cyx[1:3, 1:3] = affine[0:2, 0:2]  # Copy linear part
    affine_cyx[1:3, 3] = affine[0:2, 2]  # Copy translation part
    return affine_cyx


def get_arguments():
    """
    Parses and checks command line arguments, and provides an help text.
    Assumes 5 and returns 5 positional command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Aligns CODEX to Xenium and saves SpatialData objects"
    )
    parser.add_argument("xenium_folder", help="path to the xenium folder")
    parser.add_argument("codex_tiff", help="path to the CODEX image in tiff format")
    parser.add_argument(
        "codex_channels", help="path to the txt file with the list of codex channels"
    )
    parser.add_argument("bounding_boxes", help="path to the bounding boxes csv file")
    parser.add_argument("output_folder", help="path to the output folder")
    args = parser.parse_args()
    return (
        args.xenium_folder,
        args.codex_tiff,
        args.codex_channels,
        args.bounding_boxes,
        args.output_folder,
    )


def get_crops(index, bounding_boxes, xenium, codex_img, xenium_img):
    """
    Crops the aligned SpatialData object, the codex and morphology focus images
    """
    max_x = bounding_boxes.loc[index].x_max
    max_y = bounding_boxes.loc[index].y_max
    min_x = bounding_boxes.loc[index].x_min
    min_y = bounding_boxes.loc[index].y_min
    crop_codex_img = codex_img[:, min_y:max_y, min_x:max_x]  # Codex
    crop_xenium_img = xenium_img[:, min_y:max_y, min_x:max_x]  # Xenium
    cropped_xenium = sd.bounding_box_query(
        xenium,
        axes=("x", "y"),
        max_coordinate=[max_x, max_y],
        min_coordinate=[min_x, min_y],
        target_coordinate_system="global",
    )
    cropped_xenium["cell_labels"] = cropped_xenium["cell_labels"].chunk(
        chunks={"y": 1024, "x": 1024}
    )
    # Codex, Xenium mf, Xenium data
    return crop_codex_img, crop_xenium_img, cropped_xenium


def process_crops(
    crop_codex, crop_xmf, cropped_xenium, codex_channels, xmf_channels, scale_factor
):
    """
    Align the cropped codex and morphology focus images, and adds them to the SpatialData object
    """
    crop_cx_dna = preprocess_reg(crop_codex[0, ...], scale_factor)
    crop_mf_dna = preprocess_reg(crop_xmf[0, ...], scale_factor)

    crop_affine_cyx = get_align_matrix(crop_cx_dna, crop_mf_dna, scale_factor)
    rrcx = dask_image.ndinterp.affine_transform(
        crop_codex, crop_affine_cyx, output_shape=crop_codex.shape
    )

    crop_translation = sd.transformations.get_transformation(
        cropped_xenium["cell_labels"]
    )

    codex = Image2DModel()
    codex = codex.parse(
        rrcx,
        c_coords=codex_channels,
        scale_factors=[2, 2, 2, 2],
        transformations={"global": crop_translation},
    )
    Image2DModel().validate(codex)
    cropped_xenium["codex"] = codex

    xenium_mf = Image2DModel()
    xenium_mf = xenium_mf.parse(
        crop_xmf,
        c_coords=xmf_channels,
        scale_factors=[2, 2, 2, 2],
        transformations={"global": crop_translation},
    )
    Image2DModel().validate(xenium_mf)
    cropped_xenium["morphology_focus"] = xenium_mf

    return cropped_xenium


def save_crop(index, bounding_boxes, cropped_xenium, out_folder):
    """
    Writes "{output_folder}/core-{int(bboxes.loc[index].core_id)}.zarr"
    """
    try:
        cropped_xenium.write(
            f"{out_folder}/core-{int(bounding_boxes.loc[index].core_id)}.zarr",
            overwrite=True,
            consolidate_metadata=True,
        )
    except AttributeError as err:
        print(err)


if __name__ == "__main__":
    (
        xenium_folder,
        codex_tiff,
        codex_channels_path,
        bounding_boxes_path,
        output_folder,
    ) = get_arguments()

    # Builds the path to the morphology focus images
    xenium_morphology_path = (
        xenium_folder + "/morphology_focus/morphology_focus_0000.ome.tif"
    )

    FIRST_SCALE_FACTOR = 8
    SECOND_SCALE_FACTOR = 2

    # Read in the bounding boxes for each core
    # core_id	x_min	x_max	y_min	y_max
    # coordinates in nm
    print("Reading bounding boxes")
    bboxes = pd.read_csv(bounding_boxes_path)
    bboxes.iloc[:, 1:] = (bboxes.iloc[:, 1:] / 0.212).astype(int)  # 1px = 212nm
    bboxes.iloc[:, 0] = bboxes.iloc[:, 0].astype(int)
    total_cores = len(bboxes)

    # Read in:
    # - Transcripts
    # - Cell labels
    # - Cell boundaries
    # - Feature matrix
    print("Reading Xenium Data")
    xenium_data = spatialdata_io.xenium(
        xenium_folder,
        cells_boundaries=True,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=True,
        nucleus_labels=False,
        transcripts=True,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        cells_table=True,
        n_jobs=8,
    )

    # Clean up the transcripts
    # Not transcripts to be discarded
    print("Cleaning up Xenium Transcripts")
    not_genes = [
        "deprecated_codeword",
        "negative_control_probe",
        "genomic_control_probe",
        "negative_control_codeword",
        "unassigned_codeword",
    ]
    # Columns to be discarded from the transcript table
    columns_to_drop = [
        "z",
        "fov_name",
        "nucleus_distance",
        "codeword_index",
        "codeword_category",
        "is_gene",
        "transcript_id",
        "overlaps_nucleus",
    ]
    # Get rid of unwanted rows and columns
    xenium_data["transcripts"] = xenium_data["transcripts"][
        ~xenium_data["transcripts"].isin(not_genes)
    ]
    xenium_data["transcripts"] = xenium_data["transcripts"].drop(
        columns=columns_to_drop
    )
    xenium_data["transcripts"] = xenium_data["transcripts"][
        xenium_data["transcripts"].feature_name.notnull()
    ]
    # Repartitionon the dask array for efficiency
    print("Repartitioning Xenium Transcripts")
    xenium_data["transcripts"] = xenium_data["transcripts"].repartition(
        partition_size="100MB"
    )

    # Write out and then read back in the repartitioned SpatialData object
    xenium_rewrite_folder = tempfile.TemporaryDirectory()
    xenium_rewrite_path = xenium_rewrite_folder.name + "/" + "sd.zarr"
    print(f"Writing out repartitioned data to {xenium_rewrite_path}")
    xenium_data.write(xenium_rewrite_path, overwrite=True, consolidate_metadata=True)
    # Use the new optimized SpatialData object
    print("Reading back repartitioned data")
    xenium_data = sd.read_zarr(xenium_rewrite_path)

    # Read the codex image converted to zarr as:
    print("Reading CODEX image")
    cx_img = tifffile.imread(codex_tiff, aszarr=True)
    cx_img = zarr.open(cx_img, mode="r")
    cx_img = dask.array.from_zarr(cx_img[0])
    cx_img = np.flip(cx_img, (1, 2))
    # Read the CODEX channel list
    print("Reading CODEX channel list")
    with open(codex_channels_path, encoding="utf-8") as cc_file:
        cx_channels = [c.strip() for c in cc_file.readlines()]

    # Read one of the xenium morphology focus images as zarr:
    print("Reading xenium morphology focus")
    mf_channels = ["DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin"]
    mf_img = tifffile.imread(xenium_morphology_path, aszarr=True)
    mf_img = zarr.open(mf_img, mode="r")
    mf_img = dask.array.from_zarr(mf_img)

    # First round for a coarse registration, allows to crop the CODEX using Xenium cooordinates
    # Downscaling CODEX DAPI
    print("Reading CODEX preprocessing")
    dna_codex = preprocess_reg(cx_img[0, ...], FIRST_SCALE_FACTOR)
    # Downscaling Xenium Morphology focus DAPI
    print("Reading xenium morphology focus preprocessing")
    dna_mf = preprocess_reg(mf_img[0, ...], FIRST_SCALE_FACTOR)
    # Get the affine transform and perform the registration
    print("Getting registration matrix")
    aff_cyx = get_align_matrix(dna_codex, dna_mf, scale_factor=FIRST_SCALE_FACTOR)
    print("Coarse registration")
    rcx_img = dask_image.ndinterp.affine_transform(
        cx_img,
        aff_cyx,
        output_shape=(cx_img.shape[0], *mf_img[0, ...].shape),
        output_chunks=(1, 1024, 1024),
    )

    # Process each core:
    for i in range(0, len(bboxes)):
        print(f"Processing core {i}/{total_cores}")
        # Crop the images and the Xenium data
        print("Cropping")
        crop_cx, crop_mf, crop_xenium = get_crops(
            i, bboxes, xenium_data, rcx_img, mf_img
        )
        # Realign the images, add them to the cropped object, and fix their transforms
        print("Registering crop")
        crop_xenium = process_crops(
            crop_cx,
            crop_mf,
            crop_xenium,
            cx_channels,
            mf_channels,
            scale_factor=SECOND_SCALE_FACTOR,
        )
        print("Saving crop")
        save_crop(i, bboxes, crop_xenium, output_folder)
        print(f"Processed core {i+1}")

    # Remove the repartitioned SpatialData object
    print("Cleanup")
    xenium_rewrite_folder.cleanup()
    print("Done")
