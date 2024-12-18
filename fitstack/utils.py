"""Utililites to prepare the data for the fit."""

import logging
import os
import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.fftpack import next_fast_len

from caput import misc
from draco.util import tools

from . import containers

logger = logging.getLogger(__name__)


def covariance(a, corr=False):
    """Calculate the sample covariance over mock catalogs or power spectra.

    Parameters
    ----------
    a : np.ndarray[nmock, nx, ...]
        Array of mock data.
    corr : bool
        Return the correlation matrix instead of the covariance matrix.
        Default is False.

    Returns
    -------
    cov : np.ndarray[nx, nx,  ...]
        The sample covariance matrix (or correlation matrix).
    """

    am = a - np.mean(a, axis=0)

    cov = np.sum(am[:, np.newaxis, :] * am[:, :, np.newaxis], axis=0) / float(
        am.shape[0] - 1
    )

    if corr:
        diag = np.diag(cov)
        cov = cov * tools.invert_no_zero(
            np.sqrt(diag[np.newaxis, :] * diag[:, np.newaxis])
        )

    return cov


def unravel_covariance(cov, npol, nx):
    """Separate the covariance matrix into sub-arrays based on polarisation.

    Parameters
    ----------
    cov : np.ndarray[npol * nx, npol * nx]
        Covariance matrix.
    npol : int
        Number of polarisations.
    nx : int
        Number of frequencies or k bins.

    Returns
    -------
    cov_by_pol : np.ndarray[npol, npol, nx, nx]
        Covariance matrix reformatted such that cov_by_pol[i,j]
        gives the covariance between polarisation i and j as
        a function of frequency offset or k.
    """

    cov_by_pol = np.zeros((npol, npol, nx, nx), dtype=cov.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nx, (aa + 1) * nx)

        for bb in range(npol):

            slc_bb = slice(bb * nx, (bb + 1) * nx)

            cov_by_pol[aa, bb] = cov[slc_aa, slc_bb]

    return cov_by_pol


def ravel_covariance(cov_by_pol):
    """Collapse the covariance matrix over the polarisation axes.

    Parameters
    ----------
    cov_by_pol : np.ndarray[npol, npol, nx, nx]
        Covariance matrix as formatted by the unravel_covariance method.

    Returns
    -------
    cov : np.ndarray[npol * nx, npol * nx]
        The covariance matrix flattened into the format required for
        inversion and subsequent likelihood computation.
    """

    npol, _, nx, _ = cov_by_pol.shape
    ntot = npol * nx

    cov = np.zeros((ntot, ntot), dtype=cov_by_pol.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nx, (aa + 1) * nx)

        for bb in range(npol):

            slc_bb = slice(bb * nx, (bb + 1) * nx)

            cov[slc_aa, slc_bb] = cov_by_pol[aa, bb]

    return cov


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def shift_and_convolve(freq, template, offset=0.0, kernel=None):
    """Shift a stacking template and (optionally) convolve with a kernel.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency offset in MHz.
    template : np.ndarray[..., nfreq]
        Template for the signal.
    offset : float
        Central frequency offset in MHz.
    kernel : np.ndarray[..., nfreq]
        Kernel to convolve with the template.

    Returns
    -------
    template_sc : np.ndarray[..., nfreq]
        Template after shifting by offset and convolving with kernel.
    """

    # Determine the size of the fft needed for convolution
    nfreq = freq.size
    assert nfreq == template.shape[-1]

    size = nfreq if kernel is None else nfreq + kernel.shape[-1] - 1
    fsize = next_fast_len(int(size))
    fslice = slice(0, int(size))

    # Determine the delay corresponding to the frequency offset
    df = np.abs(freq[1] - freq[0]) * 1e6
    tau = np.fft.rfftfreq(fsize, d=df) * 1e6

    shift = np.exp(-2.0j * np.pi * tau * offset)

    # Take the fft and apply the delay
    fft_model = np.fft.rfft(template, fsize, axis=-1) * shift

    # Multiply by the fft of the kernel (if provided)
    if kernel is not None:
        fft_model *= np.fft.rfft(kernel, fsize, axis=-1)

    # Perform the inverse fft and center appropriately
    model = np.fft.irfft(fft_model, fsize, axis=-1)[..., fslice].real
    model = _centered(model, template.shape)

    return model


def combine_pol(cnt):
    """Perform a weighted sum of the XX and YY polarisations.

    Parameters
    ----------
    cnt : FrequencyStackByPol, MockFrequencyStackByPol, Powerspec1D, MockPowerspec1D
        Input container.

    Returns
    -------
    z : np.ndarray
        The weighted sum of the relevant dataset for the XX and YY polarisations.
    wz : np.ndarray
        The sum of the weights for the XX and YY polarisations.
    x : np.ndarray
        The weighted average of the independent coordinate (frequency lag or k)
        for the XX and YY polarisations.
    """

    dset = "stack" if isinstance(cnt, containers.FrequencyStackByPol) else "ps1D"

    pol = list(cnt.pol)

    # If operating on power spectra, check that ordering of k-bin centers is
    # identical for the two polarizations. (Otherwise, we shouldn't combine them.)
    if dset == "ps1D":
        isort = np.argsort(cnt.k1D)
        ax = list(cnt.k1D.attrs["axis"]).index("pol")
        slc_XX = (slice(None),) * ax + (pol.index("XX"),)
        slc_YY = (slice(None),) * ax + (pol.index("YY"),)
        if not np.allclose(isort[slc_XX], isort[slc_YY]):
            raise RuntimeError(
                "Power spectrum k bins have different ordering "
                "for different polarizations, so can't combine"
            )

    y = cnt[dset][:]
    w = (
        cnt["weight"][:]
        if dset == "stack"
        else tools.invert_no_zero(cnt["ps1D_var"][:])
    )

    ax = list(cnt[dset].attrs["axis"]).index("pol")

    flag = np.zeros_like(w)
    for pstr in ["XX", "YY"]:
        pp = pol.index(pstr)
        slc = (slice(None),) * ax + (pp,)
        flag[slc] = 1.0

    w = flag * w

    wz = np.sum(w, axis=ax)

    z = np.sum(w * y, axis=ax) * tools.invert_no_zero(wz)

    if dset == "stack":
        # Frequencies are identical for XX and YY, so no average needed
        x = cnt.freq
    else:
        x = cnt.k1D[:]
        x = np.sum(w * x, axis=ax) * tools.invert_no_zero(wz)

    return z, wz, x


def initialize_pol(cnt, pol=None, combine=False):
    """Select the data for the desired polarisations.

    Parameters
    ----------
    cnt : FrequencyStackByPol or Powerspec1D
        Container with stack or power spectrum.
    pol : list of str
        The polarisations to select.  If not provided,
        then ["XX", "YY"] is assumed.
    combine : bool
        Add an element to the polarisation axis that is
        the weighted sum of XX and YY.

    Returns
    -------
    data : np.ndarray[..., npol, nx]
        The stack or power spectrum dataset for the selected
        polarisations.
        If combine is True, there will be an additional
        element that is the weighted sum of the
        stack/power spectrum for
        the "XX" and "YY" polarisations.
    weight : np.ndarray[..., npol, nx]
        The weight dataset for the selected polarisations.
        If combine is True, there will be an additional
        element that is the sum of the weights for
        the "XX" and "YY" polarisations.
    cpol : list of str
        List of polarizations in output arrays.
    x : np.ndarray[..., npol, nx]
        Frequencies or k values for the selected polarizations.
    """

    if pol is None:
        pol = ["XX", "YY"]

    cpol = list(cnt.pol)
    ipol = np.array([cpol.index(pstr) for pstr in pol])

    num_cpol = ipol.size

    num_pol = num_cpol + int(combine)

    if isinstance(cnt, containers.FrequencyStackByPol):
        dset = "stack"
    else:
        dset = "ps1D"

    ax = list(cnt[dset].attrs["axis"]).index("pol")
    shp = list(cnt[dset].shape)
    shp[ax] = num_pol

    data = np.zeros(shp, dtype=cnt[dset].dtype)
    weight = np.zeros(shp, dtype=cnt[dset].dtype)
    x = np.zeros(shp, dtype=cnt[dset].dtype)

    slc_in = (slice(None),) * ax + (ipol,)
    slc_out = (slice(None),) * ax + (slice(0, num_cpol),)

    data[slc_out] = cnt[dset][slc_in]
    weight[slc_out] = (
        cnt["weight"][slc_in]
        if dset == "stack"
        else tools.invert_no_zero(cnt.datasets["ps1D_var"][slc_in])
    )
    if dset == "stack":
        x[slc_out] = cnt.freq[..., :]
    else:
        x[slc_out] = cnt.k1D[:]

    if combine:
        slc_out = (slice(None),) * ax + (-1,)
        temp, wtemp, xtemp = combine_pol(cnt)
        data[slc_out] = temp
        weight[slc_out] = wtemp
        x[slc_out] = xtemp
        cpol.append("I")

    return data, weight, cpol, x


def average_data(cnt, pol=None, combine=True, sort=True):
    """Calculate the mean and variance of a set of stacks or power spectra.

    Parameters
    ----------
    cnt : MockFrequencyStackByPol or MockPowerspec1D
        Container with stacks or power spectra to average.
    pol : list of str
        The polarisations to select.  If not provided,
        then ["XX", "YY"] is assumed.
    combine : bool
        Add an element to the polarisation axis that is
        the weighted sum of XX and YY.  Default is True.
    sort : bool
        Sort the frequency offset or k axis in ascending order.
        Default is True.

    Returns
    -------
    avg : FrequencyStackByPol or Powerspec1D
        Container that has collapsed over the mock axis.
        The stack or ps1D dataset contains the mean. For stacks,
        the weight dataset contains the inverse variance,
        while for power spectra, the ps1D_var dataset contains
        the variance.
    """

    darr, _, dpol, dx = initialize_pol(cnt, pol=pol, combine=combine)
    ndata = darr.shape[0]

    # If requested, sort by freq/k
    if sort:
        isort = np.argsort(dx, axis=-1)
        dx = np.take_along_axis(dx, isort, axis=-1)
        darr = np.take_along_axis(darr, isort, axis=-1)

    # Make new container with mean and variance over mock axis
    if isinstance(cnt, containers.MockFrequencyStackByPol):
        avg = containers.FrequencyStackByPol(
            pol=np.array(dpol), freq=cnt.freq, attrs_from=cnt
        )
        avg.stack[:] = np.mean(darr, axis=0)
        avg.weight[:] = tools.invert_no_zero(np.var(darr, axis=0))
    else:
        avg = containers.Powerspec1D(
            pol=np.array(dpol), k=cnt.index_map["k"], attrs_from=cnt, distributed=False
        )
        avg.k1D[:] = np.mean(dx, axis=0)
        avg.ps1D[:] = np.mean(darr, axis=0)
        avg.ps1D_var[:] = np.var(darr, axis=0)

    avg.attrs["num"] = ndata

    return avg


def load_pol(filename, pol=None):
    """Load a file, down-selecting along the polarisation axis.

    This is a wrapper for the from_file method of
    container.BaseContainer that first opens the file
    using h5py to determines the appropriate container type
    and indices into the polarisation axis.

    Parameters
    ----------
    filename : str
        Name of the file.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"].

    Returns
    -------
    out : subclass of containers.BaseContainer
        File in the appropriate container with
        the requested polarisations.
    """

    if pol is None:
        pol = ["XX", "YY"]

    pol = np.atleast_1d(pol)

    with h5py.File(filename, "r") as handler:
        container_path = handler.attrs["__memh5_subclass"]
        fpol = list(handler["index_map"]["pol"][:].astype(str))

    ipol = np.array([fpol.index(pstr) for pstr in pol])

    Container = misc.import_class(container_path)

    return Container.from_file(filename, pol_sel=ipol, distributed=False)


def load_mocks(mocks, pol=None):
    """Load the mock catalog stacks/noise power spectra.

    Parameters
    ----------
    mocks : list of str; container; list of containers; or glob
        Set of stacks on mock catalogs or noise power spectra.
        This can either be a MockFrequencyStackByPol container;
        a MockPowerspec1D container; a list of
        FrequencyStackByPol or Powerspec1D containers;
        or a filename or list of filenames that
        hold these types of containers and will be loaded from disk.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"].

    Returns
    -------
    out : MockFrequencyStackByPol or MockPowerspec1D
        All mock catalogs or power spectra in a single container.
    """

    if pol is None:
        pol = ["XX", "YY"]

    pol = np.atleast_1d(pol)

    if isinstance(mocks, containers.MockFrequencyStackByPol) or isinstance(
        mocks, containers.MockPowerspec1D
    ):

        if not np.array_equal(mocks.pol, pol):
            raise RuntimeError(
                "The mock catalogs/power spectra that were provided have "
                "incorrect polarisations."
            )

        out = mocks

    else:

        if isinstance(mocks, str):
            mocks = sorted(glob.glob(mocks))

        temp = []
        for mfile in mocks:
            if isinstance(mfile, (str, Path)):
                temp.append(load_pol(mfile, pol=pol))
            else:
                if not np.array_equal(mfile.pol, pol):
                    raise RuntimeError(
                        "The mock catalogs/power spectra that were provided have "
                        "incorrect polarisations."
                    )
                temp.append(mfile)

        nmocks = [
            mock.index_map["mock"].size if "mock" in mock.index_map else 1
            for mock in temp
        ]

        boundaries = np.concatenate(([0], np.cumsum(nmocks)))

        if isinstance(temp[0], containers.FrequencyStackByPol):
            out = containers.MockFrequencyStackByPol(
                mock=np.arange(boundaries[-1], dtype=int),
                axes_from=temp[0],
                attrs_from=temp[0],
            )
        else:
            out = containers.MockPowerspec1D(
                mock=np.arange(boundaries[-1], dtype=int),
                axes_from=temp[0],
                attrs_from=temp[0],
            )

        for mm, (mock, nm) in enumerate(zip(temp, nmocks)):

            if nm > 1:
                slc_out = slice(boundaries[mm], boundaries[mm + 1])
            else:
                slc_out = boundaries[mm]

            if isinstance(temp[0], containers.FrequencyStackByPol):
                out.stack[slc_out] = mock.stack[:]
                out.weight[slc_out] = mock.weight[:]
            else:
                out.ps1D[slc_out] = mock.ps1D[:]
                out.ps1D_error[slc_out] = mock.ps1D_error[:]
                out.ps1D_var[slc_out] = mock.ps1D_var[:]

        if isinstance(temp[0], containers.Powerspec1D):
            out.k1D[:] = mock.k1D[:]

    return out


def find_file(search):
    """Find the most recent file matching a glob string.

    Parameters
    ----------
    search : str
        Glob string to search.

    Returns
    -------
    filename : str
        Most recently modified file that matches the search.
    """

    files = glob.glob(search)
    files.sort(reverse=True, key=os.path.getmtime)

    nfiles = len(files)

    if nfiles == 0:
        raise ValueError(f"Could not find file {search}")

    elif nfiles > 1:
        ostr = "\n".join([f"({ii+1}) {ff}" for ii, ff in enumerate(files)])
        logger.warning(
            f"Found {nfiles} files that match search criteria.  " "Using (1):\n" + ostr
        )

    return files[0]
