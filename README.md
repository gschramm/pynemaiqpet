# pynemaiqpet

python routines to analyze NEMA image quality phantom scans

## Authors

Georg Schramm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Installation

We recommend to use the anaconda python distribution and to create a
conda virtual environment for pynemaiqpet.

The installation consists of three steps:

1. (optional, if not done) Installation of anaconda / miniforge python distribution
   to get the conda / mamba package manager
2. Creation of the conda virtual environment including the `pynemaiqpet`
   python package and command line tools from `conda-forge`

### Installation of miniforge

You can either install anaconda, or use the community driven miniforge
distribution [here](https://github.com/conda-forge/miniforge)

### Creation of the virtual conda environment and installation of pynemaiqpet

You can create a virtual conda environment containing the `pynemaiqpet`
python package and command line tools via:

```
conda create -c conda-forge -n pynemaiqpet pynemaiqpet
```

After installation, activate the environment via

```

conda activate pynemaiqpet
```

To test your installation you can execute the following in python

```python
import pynemaiqpet
print(pynemaiqpet.__version__)
print(pynemaiqpet.__file__)
```

or by displaying the help of the command line tool

```
pynemaiqpet_wb_nema_iq -h
```

## Run demos

To analyze a PET reconstruction of the NEMA WB phantom stored in dicom in
the folder `my_pet_recon` using the command line tool, you can execute:

```
pynemaiqpet_wb_nema_iq my_pet_recon --output_dir my_pet_recon_results --show --verbose
```

You can also apply an additional isotropic Gaussian post filter to the reconstructed
image, before running the analysis using the `--fwhm_mm` command line argument:

```
pynemaiqpet_wb_nema_iq my_pet_recon --output_dir my_pet_recon_results_5mm_fwhm_gauss --show --verbose --fwhm_mm 5.0
```

**Note:** This github repository contains two example NEMA recons in the [data subfolder](./data).
