# **Retinexformer with Halo Detection and Removal**

## License

This project is based on [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), created by Yuanhao Cai and licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.

RetinexFormer is a state-of-the-art image enhancement framework designed to improve the quality of low-light images by removing noise and corruption. The model is based on a single-stage Retinex approach integrated with an Illuminance-guided transformer. This framework has shown impressive results in low-light image enhancement, addressing common issues such as noise due to low signal-to-noise ratio (SNR) and color distortions caused by uneven lighting or sensor limitations.

However, after conducting extensive comparisons with other models, RetinexFormer was found to be the best-performing baseline, though it still exhibited artifacts, particularly halo effects around sharp edges in enhanced images. As a result, this repository adds a feature for automatic halo detection and removal to further improve the quality of the output.


# Retinexformer

We have updated the code in Retinexformer_arch.py to run on CPU instead of GPU, and modified input and target file paths for compatibility. 

Run the ipynb located at: <span style='color: green;'>drive/MyDrive/Retinexformer-master/Retinexformer.ipynb</span>. It contains commands to sucessfully execute the model and view the model output at <span style='color: green;'>drive/MyDrive/Retinexformer-master/ModelOutput</span>. These commands are also listed below.

## Run retinexfromer to view model output

1. Install Dependencies
```python
!pip install einops
!pip install natsort
```

2. Mount your Google Drive where you the repository
```
from google.colab import drive
drive.mount('/content/drive')
```

3. You can clone the repository directly into your Google Drive
```
!git clone https://github.com/ketkipatankar18/Retinexformer-master /content/drive/MyDrive/
```

4. To run the model, you can use the below code snippet
```
%cd /content
!python3 drive/MyDrive/Retinexformer-master/basicsr/models/archs/RetinexFormer_arch.py
```

You will be able to see the model generated output for 15 different images.

<p>The input images will be located at: <span style='color: green;'>drive/MyDrive/Retinexformer-master/data/NTIRE/mini_val/input/</span></p>
<p>The target\ground truth images will be located at:<span style='color: green;'>drive/MyDrive/Retinexformer-master/data/NTIRE/mini_val/target/</span></p>
<p>The model output images will be located at:<span style='color: green;'>drive/MyDrive/Retinexformer-master/ModelOutput</span></p>

On analysing the model output, the images that are corrupted with Halo artifact can then be copied to the HaloDetected folder.

# Halo Detection and Removal

This repository contains two approaches for halo removal:

**Neighborhood-based Filtering**: This method uses local neighborhoods of pixels to remove halos based on intensity thresholds.

**Neighborhood-based Filtering + Anisotropic Diffusion**: This method combines neighborhood filtering with anisotropic diffusion to refine the halo removal process.

 <p>The ipynb file containing the two approaches is present at <span style='color: green;'>drive/MyDrive/Retinexformer-master/Halo_Detection_and_Removal/Halo_Detection_and_Removal.ipynb</span></p>

 On executing the two approaches presented in this file, you will be able to view the output for each of the image you added in HaloDetected folder under <span style='color: green;'>drive/MyDrive/Retinexformer-master/Halo_Detection_and_Removal/HaloReduced_Approach1</span> and <span style='color: green;'>drive/MyDrive/Retinexformer-master/Halo_Detection_and_Removal/HaloReduced_Approach2</span>.
