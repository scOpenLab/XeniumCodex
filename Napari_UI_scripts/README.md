# Script to add widgets to Napari for interactive viewing

## Using the scripts
The scripts require 'napari' with the `napari-spatialdata` plugin (https://spatialdata.scverse.org/projects/napari/en/latest/) installed. 

To use the scripts:
1) Open Napari
2) Load the SpatialData object with `Open with Plugin` - > `OpenFolder` -> `xeniumcodex.zarr`:

  The object contains two coordinate spaces:
  - `codex`: original codex coordinates (you can ignore this)
  - `global`: Xenium + aligned codex data (this should be used for visualization)
  The `global` space should be used for visualization.

![image](https://github.com/user-attachments/assets/2a310578-5850-41d3-9043-be81c2ebf794)

3) Load the `codex` and `transcripts` layers.
   
![image](https://github.com/user-attachments/assets/4611015f-4f26-4871-87c8-888881c154e6)

5) Open the napari consloe with `Window` -> `Console` 
6) Copy-Paste the scripts in the Napari console, this will make the widgets appear on the right side.

## CODEX Channel Selector

`codex_selector.py`

Creates a new image layer from the selected CODEX channel. The colormap and range of the selected channel can then be selected from the standard Napari UI on the top left. 

![image](https://github.com/user-attachments/assets/7b082eb6-bad3-4b88-8b2c-bccfdc7e6c03)

##  Xenium Transcript Selector

`codex_selector.py`

Creates a new point layer from the selected Xenium . After selecting all points with `Ctrl + A`, their size and color can be changed from the standard napari UI.

![image](https://github.com/user-attachments/assets/93c8bb52-68fb-4f84-8af3-07779b7a3bd8)



