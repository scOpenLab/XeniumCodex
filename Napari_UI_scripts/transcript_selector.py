from magicgui import magicgui


# Setup: get the transcript layer and gene list
transcript_layer = viewer.layers["transcripts"] # Assumes the layer has been loaded
transcript_layer.features = transcript_layer.metadata["_columns_df"]
gene_list = sorted(transcript_layer.features["feature_name"].unique())

# Create the widget
@magicgui(
    call_button="Filter Genes",
    gene={"label": "Select Genes", "widget_type": "Select", "choices": gene_list}
)
def multi_gene_filter_widget(layer: "napari.layers.Points", gene: list):
    coords = layer.data
    features = transcript_layer.metadata["_columns_df"]

    # Filter for selected genes
    mask = features["feature_name"].isin(gene)
    filtered_coords = coords[mask.values]
    filtered_features = features[mask]

    # Add filtered points layer
    viewer.add_points(
        filtered_coords,
        features=filtered_features,
        name=f"filtered_{'_'.join(gene)}",
        affine = transcript_layer.affine
    )

# Add to Napari dock
viewer.window.add_dock_widget(multi_gene_filter_widget, area="right")

