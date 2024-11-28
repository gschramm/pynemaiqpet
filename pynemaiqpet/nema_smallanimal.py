import math
import numpy as np
import pylab as py

import pymirc.metrics as pymr

from pymirc.image_operations import kul_aff, aff_transform

from scipy.ndimage import label, labeled_comprehension, find_objects, gaussian_filter
from scipy.ndimage import binary_erosion, median_filter, binary_closing
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

from scipy.special import erf
from scipy.integrate import quad

from scipy.signal import argrelextrema
from scipy.optimize import minimize

from lmfit import Model


# ----------------------------------------------------------------------
def cylinder_prof_integrand(eta, z, Z):
    """Integrand of the convolution of a disk convolved with a 2D Gaussian

    Parameters
    ----------
    eta : float
      integration variable

    z : float
      radial distance from center divided by sigma of Gaussian

    Z : float
      disk radius divided by sigma of Gaussian

    Returns
    -------
    float
    """
    return math.exp(-0.5 * eta**2) * (
        erf((math.sqrt(Z**2 - eta**2) - z) / math.sqrt(2))
        + erf((math.sqrt(Z**2 - eta**2) + z) / math.sqrt(2))
    )


# ----------------------------------------------------------------------
def cylinder_prof(z, Z):
    """Normalized radial profile of a disk convolved with a 2D Gaussian

    Parameters
    ----------
    z : float
      radial distance from center divided by sigma of Gaussian

    Z : float
      disk radius divided by sigma of Gaussian

    Returns
    -------
    float

    Note
    ----
    There is no analytic expression for the convolution.
    The profile is numerically integrated using quad() from scipy.integrate
    """
    return quad(cylinder_prof_integrand, 0, Z, args=(z, Z))[0] / math.sqrt(2 * math.pi)


# ----------------------------------------------------------------------
def cylinder_profile(r, S=1.0, R=1.0, fwhm=1.0):
    """Radial profile of a disk convolved with a 2D Gaussian

    Parameters
    ----------
    r : 1D numpy float array
      radial distance from center

    S : float
      signal in the disk

    R : float
      radius of the disk

    fwhm : float
      FWHM of the Gaussian smoothing kernel

    Returns
    -------
    1D numpy float array
    """
    sig = fwhm / 2.35

    cp = np.frompyfunc(cylinder_prof, 2, 1)

    return S * cp(r / sig, R / sig).astype(float)


# ----------------------------------------------------------------------
def fit_nema_2008_cylinder_profiles(
    vol,
    voxsize,
    Rrod_init=[2.5, 2, 1.5, 1, 0.5],
    fwhm_init=1.5,
    S_init=1,
    fix_S=True,
    fix_R=False,
    fix_fwhm=False,
    nrods=4,
    phantom="standard",
):
    """Fit the radial profiles of the rods in a nema 2008 small animal PET phantom

    Parameters
    ----------
    vol : 3D numpy float array
      containing the image

    voxsize : 3 element 1D numpy array
      containing the voxel size

    Rrod_init : list or 1D numpy array of floats, optional
      containing the initial values of the rod radii

    S_init, fwhm_init: float, optional
      initial values for the signal and the FWHM in the fit

    fix_S, fix_R, fix_fwhm : bool, optional
      whether to keep the initial values of signal, radius and FWHM fixed during the fix

    nrods: int, optional
      number of rods to fit

    phantom : string
      phantom version ('standard' or 'mini')

    Returns
    -------
    a list of lmfit fit results

    Note
    ----

    The axial direction should be the right most direction in the 3D numpy array.
    The slices containing the rods are found automatically and summed.
    In the summed image, all rods (disks) are segmented followed by a fit
    of the radial profile.
    """
    roi_vol = nema_2008_small_animal_pet_rois(vol, voxsize, phantom=phantom)

    rod_bbox = find_objects(roi_vol == 4)

    # find the rods in the summed image
    sum_img = vol[:, :, rod_bbox[0][2].start : rod_bbox[0][2].stop].mean(2)

    label_img, nlab = label(sum_img > 0.1 * sum_img.max())
    labels = np.arange(1, nlab + 1)
    # sort the labels according to volume
    npix = labeled_comprehension(sum_img, label_img, labels, len, int, 0)
    sort_inds = npix.argsort()[::-1]
    labels = labels[sort_inds]
    npix = npix[sort_inds]

    # ----------------------------------------------------------------------
    ncols = 2
    nrows = int(np.ceil(nrods / ncols))
    fig, ax = py.subplots(
        nrows, ncols, figsize=(12, 7 * nrows / 2), sharey=True, sharex=True
    )

    retval = []

    for irod in range(nrods):
        rod_bbox = find_objects(label_img == labels[irod])

        rod_bbox = [
            (
                slice(rod_bbox[0][0].start - 2, rod_bbox[0][0].stop + 2),
                slice(rod_bbox[0][1].start - 2, rod_bbox[0][1].stop + 2),
            )
        ]

        rod_img = sum_img[rod_bbox[0]]
        com = np.array(center_of_mass(rod_img))

        x0 = (np.arange(rod_img.shape[0]) - com[0]) * voxsize[0]
        x1 = (np.arange(rod_img.shape[1]) - com[1]) * voxsize[1]

        X0, X1 = np.meshgrid(x0, x1, indexing="ij")
        RHO = np.sqrt(X0**2 + X1**2)

        rho = RHO.flatten()
        signal = rod_img.flatten()

        # sort the values according to rho
        sort_inds = rho.argsort()
        rho = rho[sort_inds]
        signal = signal[sort_inds]

        pmodel = Model(cylinder_profile)
        params = pmodel.make_params(S=S_init, R=Rrod_init[irod], fwhm=fwhm_init)

        if fix_S:
            params["S"].vary = False
        if fix_R:
            params["R"].vary = False
        if fix_fwhm:
            params["fwhm"].vary = False

        fitres = pmodel.fit(signal, r=rho, params=params)
        retval.append(fitres)
        fit_report = fitres.fit_report()

        iplot = np.unravel_index(irod, ax.shape)
        ax[iplot].plot(rho, signal, "k.")

        rfit = np.linspace(0, rho.max(), 100)
        ax[iplot].plot(rfit, fitres.eval(r=rfit), "r-")
        ax[iplot].text(
            0.99,
            0.99,
            fit_report,
            fontsize=6,
            transform=ax[iplot].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            backgroundcolor="white",
            bbox={"pad": 0, "facecolor": "white", "lw": 0},
        )
        ax[iplot].grid()

    for axx in ax[-1, :]:
        axx.set_xlabel("R (mm)")
    for axx in ax[:, 0]:
        axx.set_ylabel("signal")

    fig.tight_layout()
    fig.show()

    return retval


# ----------------------------------------------------------------------
def nema_2008_small_animal_pet_rois(
    vol, voxsize, lp_voxel="max", rod_th=0.15, phantom="standard"
):
    """generate a label volume indicating the ROIs needed in the analysis of the
        NEMA small animal PET IQ phantom

    Parameters
    ----------
    vol : 3D numpy float array
      containing the image

    voxsize : 3 element 1D numpy array
      containing the voxel size

    lp_voxel: string, optional
      method of how to compute the ROIs around the line profiles
      in the summed images of the hot rods.
      'max' means the maximum voxels in the summed 2D image.
      anything else means use all pixels that are within the rod radius
      around the center of mass

    rod_th : float, optional
      threshold to find the rod in the summed 2D image relative to the
      mean of the big uniform region

    phantom : string
      phantom version ('standard' or 'mini')

    Returns
    -------
    a 3D integer numpy array
      encoding the following ROIs:
      1 ... ROI of the big uniform region
      2 ... first cold insert
      3 ... second cold insert
      4 ... central line profile in 5mm rod
      5 ... central line profile in 4mm rod
      6 ... central line profile in 3mm rod
      7 ... central line profile in 2mm rod
      8 ... central line profile in 1mm rod

    Note
    ----
    The rod ROIs in the summed 2D image are found by thresholding.
    If the activity in the small rods is too low, they might be missed.
    """
    roi_vol = np.zeros(vol.shape, dtype=np.uint)

    # calculate the summed z profile to place the ROIs
    zprof = vol.sum(0).sum(0)
    zprof_grad = np.gradient(zprof)
    zprof_grad[np.abs(zprof_grad) < 0.13 * np.abs(zprof_grad).max()] = 0

    rising_edges = argrelextrema(zprof_grad, np.greater, order=10)[0]
    falling_edges = argrelextrema(zprof_grad, np.less, order=10)[0]

    # if we only have 2 falling edges because the volume is cropped, we add the last slices as
    # falling edge

    if falling_edges.shape[0] == 2:
        falling_edges = np.concatenate([falling_edges, [vol.shape[2]]])

    # define and analyze the big uniform ROI
    uni_region_start_slice = rising_edges[1]
    uni_region_end_slice = falling_edges[1]
    uni_region_center_slice = 0.5 * (uni_region_start_slice + uni_region_end_slice)

    uni_roi_start_slice = int(np.floor(uni_region_center_slice - 5.0 / voxsize[2]))
    uni_roi_end_slice = int(np.ceil(uni_region_center_slice + 5.0 / voxsize[2]))

    uni_com = np.array(
        center_of_mass(vol[:, :, uni_roi_start_slice : (uni_roi_end_slice + 1)])
    )

    x0 = (np.arange(vol.shape[0]) - uni_com[0]) * voxsize[0]
    x1 = (np.arange(vol.shape[1]) - uni_com[1]) * voxsize[1]
    x2 = (np.arange(vol.shape[2]) - uni_com[2]) * voxsize[2]

    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
    RHO = np.sqrt(X0**2 + X1**2)

    uni_mask = np.zeros(vol.shape, dtype=np.uint)
    if phantom == "standard":
        uni_mask[RHO <= 11.25] = 1
    elif phantom == "mini":
        uni_mask[RHO <= 6.25] = 1
    uni_mask[:, :, :uni_roi_start_slice] = 0
    uni_mask[:, :, (uni_roi_end_slice + 1) :] = 0

    uni_inds = np.where(uni_mask == 1)
    roi_vol[uni_inds] = 1

    # define and analyze the two cold ROIs
    insert_region_start_slice = falling_edges[1]
    insert_region_end_slice = falling_edges[2]
    insert_region_center_slice = 0.5 * (
        insert_region_start_slice + insert_region_end_slice
    )

    insert_roi_start_slice = int(
        np.floor(insert_region_center_slice - 3.75 / voxsize[2])
    )
    insert_roi_end_slice = int(np.ceil(insert_region_center_slice + 3.75 / voxsize[2]))

    # sum the insert slices and subtract them from the max to find the two cold inserts
    sum_insert_img = vol[
        :, :, insert_roi_start_slice : (insert_roi_end_slice + 1)
    ].mean(2)

    ref = np.percentile(sum_insert_img, 99)
    if phantom == "standard":
        insert_label_img, nlab_insert = label(sum_insert_img <= 0.5 * ref)
    elif phantom == "mini":
        # reset pixels outside the phantom, since inserts sometimes leak into background
        tmp_inds = RHO[:, :, 0] > 9
        sum_insert_img[tmp_inds] = ref
        insert_label_img, nlab_insert = label(
            binary_erosion(sum_insert_img <= 0.5 * ref)
        )

        # add backgroud low activity ROI to be compliant with standard phantom
        insert_label_img[tmp_inds] = 3
        nlab_insert += 1

    insert_labels = np.arange(1, nlab_insert + 1)
    # sort the labels according to volume
    npix_insert = labeled_comprehension(
        sum_insert_img, insert_label_img, insert_labels, len, int, 0
    )
    insert_sort_inds = npix_insert.argsort()[::-1]
    insert_labels = insert_labels[insert_sort_inds]
    npix_insert = npix_insert[insert_sort_inds]

    for i_insert in [1, 2]:
        tmp = insert_label_img.copy()
        tmp[insert_label_img != insert_labels[i_insert]] = 0
        com_pixel = np.round(np.array(center_of_mass(tmp)))

        x0 = (np.arange(vol.shape[0]) - com_pixel[0]) * voxsize[0]
        x1 = (np.arange(vol.shape[1]) - com_pixel[1]) * voxsize[1]
        x2 = (np.arange(vol.shape[2])) * voxsize[2]

        X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
        RHO = np.sqrt(X0**2 + X1**2)

        insert_mask = np.zeros(vol.shape, dtype=np.uint)
        insert_mask[RHO <= 2] = 1
        insert_mask[:, :, :insert_roi_start_slice] = 0
        insert_mask[:, :, (insert_roi_end_slice + 1) :] = 0

        insert_inds = np.where(insert_mask == 1)
        roi_vol[insert_inds] = i_insert + 1

    # find the rod z slices
    rod_start_slice = falling_edges[0]
    rod_end_slice = rising_edges[1]
    rod_center = 0.5 * (rod_start_slice + rod_end_slice)

    rod_roi_start_slice = int(np.floor(rod_center - 5.0 / voxsize[2]))
    rod_roi_end_slice = int(np.ceil(rod_center + 5.0 / voxsize[2]))

    # sum the rod slices
    sum_img = vol[:, :, rod_roi_start_slice : (rod_roi_end_slice + 1)].mean(2)

    # label the summed image
    label_img, nlab = label(sum_img > rod_th * sum_img.max())
    labels = np.arange(1, nlab + 1)

    # sort the labels according to volume
    npix = labeled_comprehension(sum_img, label_img, labels, len, int, 0)
    sort_inds = npix.argsort()[::-1]
    labels = labels[sort_inds]
    npix = npix[sort_inds]

    # find the center for the line profiles
    for i, lab in enumerate(labels):

        if lp_voxel == "max":
            rod_sum_img = sum_img.copy()
            rod_sum_img[label_img != lab] = 0

            central_pixel = np.unravel_index(rod_sum_img.argmax(), rod_sum_img.shape)
            roi_vol[
                central_pixel[0],
                central_pixel[1],
                rod_roi_start_slice : (rod_roi_end_slice + 1),
            ] = (
                i + 4
            )
        else:
            bbox = find_objects(label_img == lab)[0]
            rod_sum_img = np.zeros(sum_img.shape)
            rod_sum_img[bbox] = sum_img[bbox]

            central_pixel = np.round(np.array(center_of_mass(rod_sum_img))).astype(
                np.int
            )

            x0 = (np.arange(rod_sum_img.shape[0]) - central_pixel[0]) * voxsize[0]
            x1 = (np.arange(rod_sum_img.shape[1]) - central_pixel[1]) * voxsize[1]
            X0, X1 = np.meshgrid(x0, x1, indexing="ij")

            dist_to_rod = np.sqrt(X0**2 + X1**2)

            for r in range(rod_roi_start_slice, (rod_roi_end_slice + 1)):
                roi_vol[..., r][dist_to_rod <= 0.5 * (5 - i)] = i + 4

    # -------------------------------------------------------
    # if we only have 4 labels (rods), we find the last (smallest) one based on symmetries
    if nlab == 4:
        roi_img = roi_vol[..., rod_roi_start_slice]

        com = center_of_mass(roi_vol == 1)
        x0 = (np.arange(sum_img.shape[0]) - com[0]) * voxsize[0]
        x1 = (np.arange(sum_img.shape[1]) - com[1]) * voxsize[1]
        X0, X1 = np.meshgrid(x0, x1, indexing="ij")
        RHO = np.sqrt(X0**2 + X1**2)

        PHI = np.arctan2(X1, X0)
        rod_phis = np.array([PHI[roi_img == x][0] for x in np.arange(4, nlab + 4)])
        PHI = ((PHI - rod_phis[3]) % (2 * np.pi)) - np.pi
        rod_phis = ((rod_phis - rod_phis[3]) % (2 * np.pi)) - np.pi

        missing_phi = ((rod_phis[3] - rod_phis[2]) % (2 * np.pi)) - np.pi

        mask = np.logical_and(np.abs(PHI - missing_phi) < 0.25, np.abs(RHO - 6.4) < 2)

        central_pixel = np.unravel_index(np.argmax(sum_img * mask), sum_img.shape)
        if lp_voxel == "max":
            roi_vol[
                central_pixel[0],
                central_pixel[1],
                rod_roi_start_slice : (rod_roi_end_slice + 1),
            ] = 8
        else:
            x0 = (np.arange(rod_sum_img.shape[0]) - central_pixel[0]) * voxsize[0]
            x1 = (np.arange(rod_sum_img.shape[1]) - central_pixel[1]) * voxsize[1]
            X0, X1 = np.meshgrid(x0, x1, indexing="ij")

            dist_to_rod = np.sqrt(X0**2 + X1**2)

            for r in range(rod_roi_start_slice, (rod_roi_end_slice + 1)):
                roi_vol[..., r][dist_to_rod <= 0.5] = 8

        nlab += 1
    # -------------------------------------------------------

    return roi_vol


# --------------------------------------------------------------------
def nema_2008_small_animal_iq_phantom(voxsize, shape, version="standard"):
    """generate a digital version of the upper part of the NEMA small animal PET
        IQ phantom that can be used to align a NEMA scan

    Parameters
    ----------
    voxsize : 3 element 1D numpy array
      containing the voxel size

    shape: 3 element tuple of integers
      shape of the volume

    version : string
      phantom version ('standard' or 'mini')

    Returns
    -------
      a 3D numpy array
    """
    x0 = (np.arange(shape[0]) - 0.5 * shape[0] - 0.5) * voxsize[0]
    x1 = (np.arange(shape[1]) - 0.5 * shape[1] - 0.5) * voxsize[1]
    x2 = (np.arange(shape[2]) - 0.5 * shape[2] - 0.5) * voxsize[2]

    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
    RHO = np.sqrt(X0**2 + X1**2)

    phantom = np.zeros(shape)
    if version == "standard":
        phantom[RHO <= 30.0 / 2] = 1
        phantom[X2 < 0] = 0
        phantom[X2 > 33.0] = 0

        RHO1 = np.sqrt((X0 - 7.5) ** 2 + X1**2)
        RHO2 = np.sqrt((X0 + 7.5) ** 2 + X1**2)

        # phantom[np.logical_and(RHO1 <= 9.2/2, X2 > 15)] = 0
        # phantom[np.logical_and(RHO2 <= 9.2/2, X2 > 15)] = 0
        phantom[np.logical_and(RHO1 <= 11 / 2, X2 > 15)] = 0
        phantom[np.logical_and(RHO2 <= 11 / 2, X2 > 15)] = 0
    elif version == "mini":
        phantom[RHO <= 20.0 / 2] = 1
        phantom[X2 < 0] = 0
        phantom[X2 > 34.0] = 0

        RHO1 = np.sqrt((X0 - 6) ** 2 + X1**2)
        RHO2 = np.sqrt((X0 + 6) ** 2 + X1**2)

        phantom[np.logical_and(RHO1 <= 7.2 / 2, X2 > 16)] = 0
        phantom[np.logical_and(RHO2 <= 7.2 / 2, X2 > 16)] = 0

    return phantom


# --------------------------------------------------------------------
def align_nema_2008_small_animal_iq_phantom(
    vol, voxsize, ftol=1e-2, xtol=1e-2, maxiter=10, maxfev=500, version="standard"
):
    """align a reconstruction of the NEMA small animal PET IQ phantom to its digital version

    Parameters
    ----------
    vol : 3D numpy float array
      containing the image

    voxsize : 3 element 1D numpy array
      containing the voxel size

    ftol, xtol, maxiter, maxfev : float / int
      parameter for the optimizer used to minimze the cost function

    version : string
      phantom version ('standard' or 'mini')

    Returns
    -------
      a 3D numpy array

    Note
    ----
    This routine can be useful to make sure that the rods in the NEMA scan are
    parallel to the axial direction.
    """
    phantom = nema_2008_small_animal_iq_phantom(voxsize, vol.shape, version=version)
    phantom *= vol[vol > 0.5 * vol.max()].mean()

    reg_params = np.zeros(6)

    # registration of down sampled volumes
    dsf = 3
    ds_aff = np.diag([dsf, dsf, dsf, 1.0])

    phantom_ds = aff_transform(
        phantom, ds_aff, np.ceil(np.array(phantom.shape) / dsf).astype(int)
    )

    res = minimize(
        pymr.regis_cost_func,
        reg_params,
        args=(phantom_ds, vol, True, True, lambda x, y: ((x - y) ** 2).mean(), ds_aff),
        method="Powell",
        options={
            "ftol": ftol,
            "xtol": xtol,
            "disp": True,
            "maxiter": maxiter,
            "maxfev": maxfev,
        },
    )

    reg_params = res.x.copy()
    # we have to scale the translations by the down sample factor since they are in voxels
    reg_params[:3] *= dsf

    res = minimize(
        pymr.regis_cost_func,
        reg_params,
        args=(phantom, vol, True, True, lambda x, y: ((x - y) ** 2).mean()),
        method="Powell",
        options={
            "ftol": ftol,
            "xtol": xtol,
            "disp": True,
            "maxiter": maxiter,
            "maxfev": maxfev,
        },
    )

    regis_aff = kul_aff(res.x, origin=np.array(vol.shape) / 2)
    vol_aligned = aff_transform(vol, regis_aff, vol.shape)

    return vol_aligned


# --------------------------------------------------------------------
def nema_2008_small_animal_iq_phantom_report(vol, roi_vol):
    """generate the report for the NEMA 2008 small animal PET IQ phantom analysis

    Parameters
    ----------
    vol : 3D numpy float array
      containing the image

    roi_vol : 3D numpy float array
      containing the ROI label image with following ROIs:
      1 ... ROI of the big uniform region
      2 ... first cold insert
      3 ... second cold insert
      4 ... central line profile in 5mm rod
      5 ... central line profile in 4mm rod
      6 ... central line profile in 3mm rod
      7 ... central line profile in 2mm rod
      8 ... central line profile in 1mm rod

    Returns
    -------
      a 3D numpy array

    Note
    ----
    The ROIs for the smaller rods are optional.
    """
    np.set_printoptions(precision=3)

    # get the ROI values of the big uniform ROI
    uni_values = vol[roi_vol == 1]
    uni_mean = uni_values.mean()
    uni_max = uni_values.max()
    uni_min = uni_values.min()
    uni_std = uni_values.std()
    uni_perc_std = 100 * uni_std / uni_mean

    print("\nuniform ROI results")
    print("------------------------------")
    print("mean ...:", "%.3f" % uni_mean)
    print("max  ...:", "%.3f" % uni_max)
    print("min  ...:", "%.3f" % uni_min)
    print("%std ...:", "%.3f" % uni_perc_std, "\n")

    # get the ROI values of the 2 cold inserts
    insert_mean = np.zeros(2)
    insert_std = np.zeros(2)

    insert1_values = vol[roi_vol == 2]
    insert_mean[0] = insert1_values.mean()
    insert_std[0] = insert1_values.std()

    insert2_values = vol[roi_vol == 3]
    insert_mean[1] = insert2_values.mean()
    insert_std[1] = insert2_values.std()

    insert_ratio = insert_mean / uni_mean
    insert_perc_std = 100 * np.sqrt(
        (insert_std / insert_mean) ** 2 + (uni_std / uni_mean) ** 2
    )

    print("\ncold insert results")
    print("------------------------------")
    print("spill over ratio ...:", insert_ratio)
    print("%std             ...:", insert_perc_std, "\n")

    # analyze the rod profiles
    nrods = int(roi_vol.max() - 3)
    lp_mean = np.zeros(nrods)
    lp_std = np.zeros(nrods)

    # find the center for the line profiles
    for i in range(nrods):
        lp_values = vol[roi_vol == i + 4]
        lp_mean[i] = lp_values.mean()
        lp_std[i] = lp_values.std()

    lp_rc = lp_mean / uni_mean
    lp_perc_std = 100 * np.sqrt((lp_std / lp_mean) ** 2 + (uni_std / uni_mean) ** 2)

    print("\nrod results")
    print("------------------------------")
    print("mean          ...:", lp_mean)
    print("recovery coeff...:", lp_rc)
    print("%std          ...:", lp_perc_std, "\n")

    np.set_printoptions(precision=None)

    res = {
        "uniform_ROI_mean": uni_mean,
        "uniform_ROI_max": uni_max,
        "uniform_ROI_min": uni_min,
        "uniform_ROI_perc_std": uni_perc_std,
        "spill_over_ratio": insert_ratio,
        "spill_over_perc_std": insert_perc_std,
        "rod_mean": lp_mean,
        "rod_recovery_coeff": lp_rc,
        "rod_perc_std": lp_perc_std,
    }

    return res


# --------------------------------------------------------------------
def get_phantom_name(vol, voxsize):
    """derive NEMA small animal phantom version from volume"""
    zprof = vol.sum(0).sum(0)
    zprof_grad = np.gradient(zprof)
    zprof_grad[np.abs(zprof_grad) < 0.1 * np.abs(zprof_grad).max()] = 0

    rising_edges = argrelextrema(zprof_grad, np.greater, order=10)[0]
    falling_edges = argrelextrema(zprof_grad, np.less, order=10)[0]

    if falling_edges.shape[0] < 2:
        falling_edges = np.concatenate([falling_edges, [vol.shape[2]]])

    center_sl = (rising_edges[1] + falling_edges[1]) // 2

    area = (
        (vol[:, :, center_sl] > 0.5 * np.percentile(vol[:, :, center_sl], 99)).sum()
        * voxsize[0]
        * voxsize[1]
    )

    if area > 450:
        phantom_name = "standard"
    else:
        phantom_name = "mini"

    return phantom_name
