# pynemaiqpet

Python routines to analyze NEMA image quality phantom scans.

## Authors

Georg Schramm

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Installation

We recommend using the Anaconda Python distribution to create a conda virtual environment for `pynemaiqpet`.

The installation consists of three steps:

1. **(Optional)** Install the Anaconda or Miniforge Python distribution to gain access to the `conda` or `mamba` package manager.
2. Create a conda virtual environment and install the `pynemaiqpet` Python package and command-line tools from `conda-forge`.

**Note:** _You can also install the package from pypi using pip install, but we recommend the
installation from conda-forge as explained below._

### Installation method 1 (recommended): Anaconda/Miniforge and conda-forge

You can either install Anaconda or use the community-driven Miniforge distribution. Find more information and downloads [here](https://github.com/conda-forge/miniforge).

To create a virtual conda environment containing the `pynemaiqpet` Python package and command-line tools, run:

```bash
conda create -c conda-forge -n pynemaiqpet pynemaiqpet
```

After installation, activate the environment by running:

```bash
conda activate pynemaiqpet
```

### Installation method 2 (not recommended): pypi and pip

```
pip install pynemaiqpet
```

### Test your installation

To test your installation, execute the following commands in Python:

```python
import pynemaiqpet
print(pynemaiqpet.__version__)
print(pynemaiqpet.__file__)
```

Alternatively, you can check the command-line tool's help page:

```bash
pynemaiqpet_wb_nema_iq -h
```

## Running Demos

To analyze a PET reconstruction of the NEMA whole-body phantom stored in DICOM format in the folder `my_pet_recon`, use the following command:

```bash
pynemaiqpet_wb_nema_iq my_pet_recon --output_dir my_pet_recon_results --show --verbose
```

To apply an additional isotropic Gaussian post-filter to the reconstructed image before analysis, use the `--fwhm_mm` argument. For example, to apply a filter with a full width at half maximum (FWHM) of 5.0 mm:

```bash
pynemaiqpet_wb_nema_iq my_pet_recon --output_dir my_pet_recon_results_5mm_fwhm_gauss --show --verbose --fwhm_mm 5.0
```

**Note:** This GitHub repository contains two example NEMA reconstructions in the [data subfolder](./data).

## Batch-processing data sets

If you need to analyze many reconstructions stored in different dicom folders,
have a look at the `vision_earl.py` or `dmi_earl.py`  
example python scripts in the [scripts subfolder](./scripts/) that show
how to do that efficiently in python.
