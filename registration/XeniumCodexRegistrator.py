import argparse
import dask
import dask_image
import skimage
import numpy as np
import cv2 as cv
import spatialdata as sd
import spatialdata_io
from spatialdata.models import Image2DModel


def get_arguments():
    """
    Parses and checks command line arguments, and provides an help text.
    Assumes 5 and returns 5 positional command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Aligns CODEX to Xenium and saves SpatialData objects"
    )
    parser.add_argument("xenium_folder", help="path to the xenium folder")
    parser.add_argument(
        "xenium_zarr",
        help="path to the xenium morphology focus DAPI image in OME-NGFF format",
    )
    parser.add_argument("codex_zarr", help="path to the CODEX image in OME-NGFF format")
    parser.add_argument(
        "codex_channels", help="path to the txt file with the list of codex channels"
    )
    parser.add_argument("output_path", help="path to the output file")
    args = parser.parse_args()
    return (
        args.xenium_folder,
        args.xenium_zarr,
        args.codex_zarr,
        args.codex_channels,
        args.output_path,
    )


if __name__ == "__main__":
    xenium_folder, xenium_zarr, codex_zarr, codex_channels, output_path = (
        get_arguments()
    )

    # Read the codex image converted to zarr as:
    print("CODEX Loading")
    cx_img = dask.array.from_zarr(codex_zarr, component="0/0")
    cx_img = cx_img.squeeze()

    # Read the CODEX channel list
    with open(codex_channels) as cc_file:
        ccoords = [c.strip() for c in cc_file.readlines()]

    # Read one of the xenium morphology focus images as zarr:
    print("Xenium Morphology Loading")
    mf_img = dask.array.from_zarr(xenium_zarr, component="0/0")
    mf_img = mf_img.max(axis=1)
    mf_img = mf_img.squeeze()

    print("Loading Xenium data")
    # Load the xenium ouptut
    aligned = spatialdata_io.xenium(
        xenium_folder,
        cells_boundaries=True,
        nucleus_boundaries=False,
        cells_as_circles=False,
        cells_labels=True,
        nucleus_labels=False,
        transcripts=True,
        morphology_mip=False,
        morphology_focus=True,
        aligned_images=False,
        cells_table=True,
        n_jobs=8,
    )

    print("Adding CODEX to the Xenium data")
    # Add the codex image to the SpatialData object
    codex = Image2DModel()
    codex_to_add = np.flip(cx_img, (1, 2))
    codex = codex.parse(
        cx_img,
        c_coords=ccoords,
        SCALE_FACTORs=[2, 2, 2],
        transformations={"codex": sd.transformations.Identity()},
    )
    Image2DModel().validate(codex)
    aligned["codex"] = codex

    print("Downscaling and preparing images for registration")
    SCALE_FACTOR = 8
    matrix = np.array([[SCALE_FACTOR, 0, 0], [0, SCALE_FACTOR, 0], [0, 0, 1]])
    # Downscaling CODEX DAPI
    output_shape = np.asarray(cx_img[0, :, :].shape)
    output_shape = (output_shape / SCALE_FACTOR).astype("uint16")

    dna_codex = dask.array.from_zarr(codex_zarr, component="0/0")
    dna_codex = cx_img.squeeze()[0, :, :]
    dna_codex = dask_image.ndinterp.affine_transform(
        dna_codex,
        matrix=matrix,
        output_shape=output_shape,
    )
    dna_codex = skimage.exposure.equalize_adapthist(dna_codex)
    dna_codex = (
        (dna_codex - np.min(dna_codex)) / (np.max(dna_codex) - np.min(dna_codex)) * 255
    )
    dna_codex = dna_codex.astype("uint8")

    # Downscaling Xenium Morphology focus DAPI
    output_shape = np.asarray(mf_img.shape)
    output_shape = (output_shape / SCALE_FACTOR).astype("uint16")
    dna_mf = dask_image.ndinterp.affine_transform(
        mf_img,
        matrix=matrix,
        output_shape=output_shape,
    )
    dna_mf = (dna_mf - np.min(dna_mf)) / (np.max(dna_mf) - np.min(dna_mf)) * 255
    dna_mf = dna_mf.astype("uint8")
    dna_mf = dna_mf.compute()

    print("Identify landmarks for registration")
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(dna_codex, None)
    kp2, des2 = sift.detectAndCompute(dna_mf, None)
    FLANN_INDEX_KDTREE = 1
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    pt1 = []
    pt2 = []
    for n, m in enumerate(good):
        pt1.append(kp1[m.queryIdx].pt)
        pt2.append(kp2[m.trainIdx].pt)
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)

    print("Preparing landmarks")
    xenium_landmarks = sd.models.ShapesModel.parse(pt2, geometry=0, radius=5)

    codex_landmarks = sd.models.ShapesModel.parse(pt1, geometry=0, radius=5)

    print("Preparing, rescaling, and applying transformation")
    affine = sd.transformations.get_transformation_between_landmarks(
        xenium_landmarks, codex_landmarks
    )
    affine.matrix[0:2, 2] *= SCALE_FACTOR
    sd.transformations.set_transformation(
        aligned["codex"],
        transformation=affine,
        to_coordinate_system="global",
        set_all=False,
        write_to_sdata=None,
    )

    print("Reducing the size of the transcript table")
    # Not transcripts to be discarded
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
    aligned["transcripts"] = aligned["transcripts"][
        ~aligned["transcripts"].isin(not_genes)
    ]
    aligned["transcripts"] = aligned["transcripts"].drop(columns=columns_to_drop)
    aligned["transcripts"] = aligned["transcripts"][
        aligned["transcripts"].feature_name.notnull()
    ]

    # Repartitionon the dask array for efficiency
    print("Repartitioning the transcript table")
    aligned["transcripts"] = aligned["transcripts"].repartition(partition_size="100MB")

    print("Writing out")
    aligned.write(output_path, overwrite=True, consolidate_metadata=True)
