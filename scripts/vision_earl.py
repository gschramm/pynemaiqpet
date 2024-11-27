import pymirc.fileio as pmf
from pathlib import Path
from scipy.ndimage import gaussian_filter
import pynemaiqpet.nema_wb as nema
import matplotlib.pyplot as plt

mdir = Path("R:/Vision_NEMAIQ/Acq_30min/")

dcm_dirs = sorted(list(mdir.glob("*APRS")))

injected_signal = 28251.9

# %%
for dcm_dir in dcm_dirs:
    print(dcm_dir)

    dcm = pmf.DicomVolume(list(dcm_dir.glob("*.dcm")))
    vol_unsmoothed = dcm.get_data()
    voxsize = dcm.voxsize

    # FWHM of Gaussian kernel to apply before analysis
    dcm_hdr = dcm.firstdcmheader

    for i_s, sm_fwhm_mm in enumerate([0, 4.0, 5.0, 7.5]):
        sm_str = f"_{sm_fwhm_mm}mm_ps"
        if sm_fwhm_mm > 0:
            print(
                f"Applying additional isotropic Gaussian post-smoothing of {sm_fwhm_mm}mm"
            )
            vol = gaussian_filter(vol_unsmoothed, sigma=sm_fwhm_mm / (2.35 * voxsize))
        else:
            vol = vol_unsmoothed

        if not dcm_hdr.ConvolutionKernel == "All-pass":
            raise ValueError("Convolution kernel is not 'All-pass'")

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
            mdir
            / "00_results"
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}.csv".replace(" ", "_")
        )

        # -------------------------------------------------------------------------------------------------
        # show the profiles

        fig = nema.show_WB_NEMA_profiles(fitres)
        fig.savefig(
            mdir
            / "00_results"
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
        # show the recoveries and EARL1 limits
        fig2 = nema.show_WB_NEMA_recoveries(sphere_results, ref_signal, earlversion=1)
        fig2.suptitle(f"{dcm.firstdcmheader.SeriesDescription}{sm_str}")
        fig2.tight_layout(pad=2)
        fig2.savefig(
            mdir
            / "00_results"
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles_EARL_1.png".replace(
                " ", "_"
            )
        )
        plt.close(fig2)

        # -------------------------------------------------------------------------------------------------
        # show the recoveries and EARL2 limits
        fig3 = nema.show_WB_NEMA_recoveries(sphere_results, ref_signal, earlversion=2)
        fig3.suptitle(f"{dcm.firstdcmheader.SeriesDescription}{sm_str}")
        fig3.tight_layout(pad=2)
        fig3.savefig(
            mdir
            / "00_results"
            / f"{dcm.firstdcmheader.SeriesDescription}{sm_str}_profiles_EARL_2.png".replace(
                " ", "_"
            )
        )
        plt.close(fig3)
