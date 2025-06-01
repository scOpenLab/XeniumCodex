import argparse
import pandas as pd
import spatialdata as sd

def get_arguments():
    """
    Parses and checks command line arguments, and provides an help text.
    Assumes 3 and returns 3 positional command line arguments:
    """
    parser = argparse.ArgumentParser(
        description="Aligns CODEX to Xenium and saves SpatialData objects"
    )
    parser.add_argument("input_folder", help="path to the xenium folder")
    parser.add_argument("bounding_boxes", help="path to the bounding boxes csv file")
    parser.add_argument("output_folder", help="path to the output folder")
    args = parser.parse_args()
    return args.input_folder, args.bounding_boxes, args.output_folder


def save_crop(index, bounds, sdata, output_path):
    """
    Crops the aligned SpatialData object and writes it out with the same structure as the original
    """
    max_x = bounds.loc[index].x_max
    max_y = bounds.loc[index].y_max
    min_x = bounds.loc[index].x_min
    min_y = bounds.loc[index].y_min
    cropped_sdata = sd.bounding_box_query(
        sdata,
        axes=("x", "y"),
        max_coordinate=[max_x, max_y],
        min_coordinate=[min_x, min_y],
        target_coordinate_system="global",
    )
    cropped_sdata["codex"] = cropped_sdata["codex"].chunk(
        chunks={"c": 1, "y": 1024, "x": 1024}
    )
    cropped_sdata["morphology_focus"] = cropped_sdata["morphology_focus"].chunk(
        chunks={"c": 1, "y": 1024, "x": 1024}
    )
    cropped_sdata["cell_labels"] = cropped_sdata["cell_labels"].chunk(
        chunks={"y": 1024, "x": 1024}
    )
    cropped_sdata.write(
        f"{output_path}/core-{int(bounds.loc[index].core_id)}.zarr",
        overwrite=True,
        consolidate_metadata=True,
    )

if __name__ == "__main__":
    input_folder, bounding_boxes, output_folder = get_arguments()

    # Use the new optimized SpatialData object
    print("Reading in again")
    aligned = sd.read_zarr(input_folder)

    # Assumes that the bounding boxes have the following CSV format:
    # - Xenium image 1px = 212nm
    # 5 columns: core_id, x_min, x_max, y_min, y_max
    print("Reading Bounding boxes")
    bboxes = pd.read_csv(bounding_boxes)
    bboxes.iloc[:, 1:] = (bboxes.iloc[:, 1:] / 0.212).astype(int)  # 1px = 212nm
    bboxes.iloc[:, 0] = bboxes.iloc[:, 0].astype(int)

    print("Writing Cropped Output")
    for i in range(0, len(bboxes)):
        save_crop(i, bboxes, aligned, output_folder)
        print(f"DONE: core {int(bboxes.loc[i].core_id)}")
