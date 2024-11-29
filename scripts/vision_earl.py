"""python script that analyses many NEMA IQ PET recons from Vision PET/CT.
The recons used different settings, but no post-smoothing was applied.
We loop over all settings and apply different level of post-smoothing before the analysis.
"""

import pymirc.fileio as pmf
from pathlib import Path
from scipy.ndimage import gaussian_filter
import pynemaiqpet.nema_wb as nema
import matplotlib.pyplot as plt

# master directory containing many subfolders each containing a dicom series
mdir = Path("R:/Vision_NEMAIQ/Acq_30min/")

# output directory for results
outdir = mdir / "00_results"
# create output directory if it does not exist
outdir.mkdir(exist_ok=True, parents=True)

# find and use all subfolders ending with "APRS"
dcm_dirs = sorted(list(mdir.glob("*APRS")))

# the signal (activity) injected in the spheres
# can be "known value", or determined from a fit of a given data set.
# this needs to be adjusted
injected_signal = 28251.9

# %%

# loop over dicom directories
for dcm_dir in dcm_dirs:
    print(dcm_dir)

    # read the dicom data
    dcm = pmf.DicomVolume(list(dcm_dir.glob("*.dcm")))
    # load the data
    vol_unsmoothed = dcm.get_data()
    # get the voxel size
    voxsize = dcm.voxsize

    # get the dicom header of the first dicom file
    dcm_hdr = dcm.firstdcmheader

    # loop over different levels of post-smoothing
    for i_s, sm_fwhm_mm in enumerate([0, 4.0, 5.0, 7.5]):
        sm_str = f"_{sm_fwhm_mm}mm_ps"
        if sm_fwhm_mm > 0:
            print(
                f"Applying additional isotropic Gaussian post-smoothing of {sm_fwhm_mm}mm"
            )
            vol = gaussian_filter(vol_unsmoothed, sigma=sm_fwhm_mm / (2.35 * voxsize))
        else:
            vol = vol_unsmoothed

        # check if the recon did not use any filter (dicom tags depends on vendor)
        if not dcm_hdr.ConvolutionKernel == "All-pass":
            raise ValueError("Convolution kernel is not 'All-pass'")

        # analysis the (post-smoothed) image
        fitres, sphere_results = nema.fit_WB_NEMA_sphere_profiles(
            vol,
            voxsize,
            sameSignal=True,
            Rfix=[18.5, 14.0, 11.0, 8.5, 6.5, 5.0],
            showBGROI=False,
            Sfix=injected_signal,
        )

        print("fit with same signal and fixed radii")
        print(sphere_results)
        sphere_results.to_csv(
            outdir
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}.csv".replace(" ", "_")
        )

        # -------------------------------------------------------------------------------------------------
        # show and save the profiles

        fig = nema.show_WB_NEMA_profiles(fitres)
        fig.savefig(
            outdir
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles.png".replace(
                " ", "_"
            ),
        )
        plt.close(fig)

        if injected_signal is None:
            ref_signal = sphere_results.signal.values[0]
        else:
            ref_signal = injected_signal

        # -------------------------------------------------------------------------------------------------
        # show and save the recoveries and EARL1 limits
        fig2 = nema.show_WB_NEMA_recoveries(sphere_results, ref_signal, earlversion=1)
        fig2.suptitle(f"{dcm.firstdcmheader.SeriesDescription}{sm_str}")
        fig2.tight_layout(pad=2)
        fig2.savefig(
            outdir
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles_EARL_1.png".replace(
                " ", "_"
            )
        )
        plt.close(fig2)

        # -------------------------------------------------------------------------------------------------
        # show and save the recoveries and EARL2 limits
        fig3 = nema.show_WB_NEMA_recoveries(sphere_results, ref_signal, earlversion=2)
        fig3.suptitle(f"{dcm.firstdcmheader.SeriesDescription}{sm_str}")
        fig3.tight_layout(pad=2)
        fig3.savefig(
            outdir
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles_EARL_2.png".replace(
                " ", "_"
            )
        )
        plt.close(fig3)
