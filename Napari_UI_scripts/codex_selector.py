from magicgui import magicgui

# Setup: get the codex layer, its transformation and channel list
codex_layer = viewer.layers["codex"]  # Assumes the layer has been loaded

# From 3D (cyx) transformation matrix to 2D (yx)
affine_transform = napari.utils.transforms.Affine( 
    affine_matrix=codex_layer.affine.affine_matrix[[1, 2, 3], :][:, [1, 2, 3]])

# Get channel names as strings
channels = list(codex_layer.metadata["adata"].var.index)
channels = [str(c) for c in channels]

# Create the widget
@magicgui(
    call_button="Filter CODEX channel",
    gene={"label": "Select Channel", "widget_type": "Select", "choices": channels}
)
def channel_filter_widget(layer: "napari.layers.Image", gene: list):
    
    gene = gene[0] # Only the first selected channel is used
    img = codex_layer.data[0].sel(c=gene) # Select from the highest resolution
    viewer.add_image(
        data=img.data,
        name=f"codex_{gene}",
        affine = affine_transform,
        blending='additive',
        rgb=False,
    )

# Add to Napari dock
viewer.window.add_dock_widget(channel_filter_widget, area="right")

