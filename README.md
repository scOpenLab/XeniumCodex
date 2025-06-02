# XeniumCodex

The repository is still a work in progress, and scripts/docs will be updated.

## Structure

The repo contains two folders one for each main step of the workflow:
- [Napari_UI_scripts](https://github.com/scOpenLab/XeniumCodex/tree/main/Napari_UI_scripts)
- [registration](https://github.com/scOpenLab/XeniumCodex/tree/main/registration)

## Workflow

# 1) Image registration and TMA core splitting

## 1.1) Making a SpatialData object with the registered CODEX image and the Xenium Data

See detailed instructions at: [registration/README.md#XeniumCodexRegistrator.py](https://github.com/scOpenLab/XeniumCodex/blob/main/registration/README.md#xeniumcodexregistratorpy)

## 1.2) Split theSpatialData object into single objects, one for each TMA core.

See detailed instructions at: [registration/README.md#RegisteredCoreSplitter.py](https://github.com/scOpenLab/XeniumCodex/blob/main/registration/README.md#registeredcoresplitterpy)

# 2) Viewing in Napari

![pastedImage](https://github.com/user-attachments/assets/b3434888-36e9-41a5-8ee5-304e7581b800)


Scripts at: Napari_UI_scripts(https://github.com/scOpenLab/XeniumCodex/tree/main/Napari_UI_scripts)

With these you can select both single-channels for the CODEX and single transcripts for the Xenium.
It's not ideal having to copy them to the Napari console, but we could turn them into something more polished later.

See detailed instructions at: [Napari_UI_scripts/README.md](https://github.com/scOpenLab/XeniumCodex/blob/main/Napari_UI_scripts/README.md)
