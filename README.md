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
1. (optional) Installation of anaconda (miniconda) python distribution
2. (optional) Creation of the conda virtual environment with all dependencies
3. Installation of the pynemaiqpet package using pip

Although optional, we highly recommend to create and use a dedicated virtual
conda python environment (steps 1 and 2).

### Installation of anaconda (miniconda)

Download and install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.

Please use the ***Python 3.x*** installer and confirm that the installer
should run 
```
conda init
``` 
at the end of the installtion process.

### Creation of the virtual conda environment

To create a virtual conda python=3.8 environment execute
```
conda create -n pynemaiqpet python=3.8 ipython
```

To test the installation of the virual environment, execute
```
conda activate pynemaiqpet
```

### Installation of the pynemaiqpet package

Activate the virtual conda environment
```
conda activate pynemaiqpet
```

Install pynemaiqpet package and all its dependecies
```
conda install -c gschramm -c conda-forge pynemaiqpet
```

To test the installation run (inside python or ipython)

```python
import pynemaiqpet
print(pynemaiqpet.__version__)
print(pynemaiqpet.__file__) 
```

## Run demos

If the installation was successful, the command line tool **pynemaiqpet_wb_nema_iq**, which allows to automatically analyze WB NEMA IQ scans from the command line, should be installed.

To list see all its command line options and the help page run
```
pynemaiqpet_wb_nema_iq -h
```
To analyze the provided demo dicom data "pet_recon_2", you e.g. run:
```
pynemaiqpet_wb_nema_iq pet_recon_2 --fwhm_mm 5 --output_dir pet_recon_2_results --show --verbose
``` 
which will read all files in the direcory "pet_recon_2", post-smooth with Gaussian with FWHM = 5mm, show the output and finally save the output into the directory "pet_recon_2_results".
