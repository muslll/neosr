import numpy as np
import pywt
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mypad(x: Tensor, pad: tuple[int, int, int, int]) -> Tensor:
    """Function to do numpy like padding with mode "periodic" on tensors.
    Only works for 2-D padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
    """
    # Vertical only
    if pad[0] == 0 and pad[1] == 0:
        xe = torch.arange(x.shape[-2])
        xe = F.pad(xe.unsqueeze(0), (pad[2], pad[3]), mode="circular").squeeze(0)
        return x[:, :, xe]
    # Horizontal only
    if pad[2] == 0 and pad[3] == 0:
        xe = torch.arange(x.shape[-1])
        xe = F.pad(xe.unsqueeze(0), (pad[0], pad[1]), mode="circular").squeeze(0)
        return x[:, :, :, xe]
    # Both
    xe_col = torch.arange(x.shape[-2])
    xe_col = torch.nn.functional.pad(
        xe_col.unsqueeze(0), (pad[2], pad[3]), mode="circular"
    ).squeeze(0)
    xe_row = torch.arange(x.shape[-1])
    xe_row = torch.nn.functional.pad(
        xe_row.unsqueeze(0), (pad[0], pad[1]), mode="circular"
    ).squeeze(0)
    i = torch.outer(xe_col, torch.ones(xe_row.shape[0]))
    j = torch.outer(torch.ones(xe_col.shape[0]), xe_row)
    return x[:, :, i, j]


def prep_filt_afb2d(
    h0_col: list[float] | list[np.ndarray] | np.ndarray,
    h1_col: list[float] | list[np.ndarray] | np.ndarray,
    h0_row: list[float] | list[np.ndarray] | np.ndarray | None = None,
    h1_row: list[float] | list[np.ndarray] | np.ndarray | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
    Returns:
        (h0_col, h1_col, h0_row, h1_row).
    """
    h0_col = np.array(h0_col[::-1]).ravel()
    h1_col = np.array(h1_col[::-1]).ravel()

    h0_row = h0_col if h0_row is None else np.array(h0_row[::-1]).ravel()
    h1_row = h1_col if h1_row is None else np.array(h1_row[::-1]).ravel()

    h0_col_tensor: Tensor = torch.tensor(h0_col, device="cuda").reshape((1, 1, -1, 1))
    h1_col_tensor: Tensor = torch.tensor(h1_col, device="cuda").reshape((1, 1, -1, 1))
    h0_row_tensor: Tensor = torch.tensor(h0_row, device="cuda").reshape((1, 1, 1, -1))
    h1_row_tensor: Tensor = torch.tensor(h1_row, device="cuda").reshape((1, 1, 1, -1))

    return h0_col_tensor, h1_col_tensor, h0_row_tensor, h1_row_tensor


def prep_filt_sfb2d(
    g0_col: Tensor,
    g1_col: Tensor,
    g0_row: Tensor | None = None,
    g1_row: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.
    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
    Returns:
        (g0_col, g1_col, g0_row, g1_row).
    """
    g0_col = g0_col.flip(dims=(0,)).flatten()
    g1_col = g1_col.flip(dims=(0,)).flatten()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    g0_col = torch.tensor(g0_col, device="cuda").reshape((1, 1, -1, 1))
    g1_col = torch.tensor(g1_col, device="cuda").reshape((1, 1, -1, 1))
    g0_row = torch.tensor(g0_row, device="cuda").reshape((1, 1, 1, -1))
    g1_row = torch.tensor(g1_row, device="cuda").reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


def afb1d_atrous(
    x: Tensor, h0: Tensor, h1: Tensor, dim: int = -1, dilation: int = 1
) -> Tensor:
    """1D analysis filter bank (along one dimension only) of an image without
    downsampling. Does the a trous algorithm.
    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
        dilation (int): dilation factor. Should be a power of 2.

    Returns
    -------
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension

    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(
            np.copy(np.array(h0).ravel()[::-1]), dtype=x.dtype, device=x.device
        )
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(
            np.copy(np.array(h1).ravel()[::-1]), dtype=x.dtype, device=x.device
        )
    L = h0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # Calculate the pad size
    L2 = (L * dilation) // 2
    pad = (0, 0, L2 - dilation, L2) if d == 2 else (L2 - dilation, L2, 0, 0)
    # ipdb.set_trace()
    x = mypad(x, pad=pad)
    return F.conv2d(x, h, groups=C, dilation=dilation)


def afb2d_atrous(
    x: Tensor, filts: tuple[Tensor, Tensor, Tensor, Tensor], dilation: int = 1
) -> Tensor:
    """Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to `afb1d_atrous`
    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
        dilation (int): dilation factor for the filters. Should be 2**level
    Returns:
        y: Tensor of shape (N, C, 4, H, W).
    """
    tensorize = [not isinstance(f, Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(h0, h1)
        else:
            h0_col = h0
            h0_row = h0.transpose(2, 3)
            h1_col = h1
            h1_row = h1.transpose(2, 3)
    elif len(filts) == 4:
        h0, h1, h0_col, h1_col = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(*filts)  # type: ignore[reportArgumentType,arg-type]
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        msg = "Unknown form for input filts"
        raise ValueError(msg)

    lohi = afb1d_atrous(x, h0_row, h1_row, dim=3, dilation=dilation)
    return afb1d_atrous(lohi, h0_col, h1_col, dim=2, dilation=dilation)


def sfb1d_atrous(
    lo: Tensor,
    hi: Tensor,
    g0: Tensor,
    g1: Tensor,
    dim: int = -1,
    dilation: int = 1,
    pad: tuple[int, int, int, int] | None = None,
) -> Tensor:
    """1D synthesis filter bank of an image tensor with no upsampling. Used for
    the stationary wavelet transform.
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, Tensor):
        g0 = torch.tensor(
            np.copy(np.array(g0).ravel()), dtype=g0.dtype, device=lo.device
        )
    if not isinstance(g1, Tensor):
        g1 = torch.tensor(
            np.copy(np.array(g1).ravel()), dtype=g1.dtype, device=lo.device
        )
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)

    # Calculate the padding size.
    # With dilation, zeros are inserted between the filter taps but not after.
    # that means a filter that is [a b c d] becomes [a 0 b 0 c 0 d].
    # TODO: test, variable was unassigned
    fsz = (L - 1) * dilation + 1

    # When conv_transpose2d is done, a filter with k taps expands an input with
    # N samples to be N + k - 1 samples. The 'padding' is really the opposite of
    # that, and is how many samples on the edges you want to cut out.
    # In addition to this, we want the input to be extended before convolving.
    # This means the final output size without the padding option will be
    #   N + k - 1 + k - 1
    # The final thing to worry about is making sure that the output is centred.
    a = fsz // 2
    b = fsz // 2 + (fsz + 1) % 2

    # pad = (0, 0, a, b) if d == 2 else (a, b, 0, 0)
    pad = (0, 0, b, a) if d == 2 else (b, a, 0, 0)
    lo = mypad(lo, pad=pad)
    hi = mypad(hi, pad=pad)

    # unpad = (fsz - 1, 0) if d == 2 else (0, fsz - 1)
    unpad = (fsz, 0) if d == 2 else (0, fsz)

    y = F.conv_transpose2d(
        lo, g0, padding=unpad, groups=C, dilation=dilation
    ) + F.conv_transpose2d(hi, g1, padding=unpad, groups=C, dilation=dilation)

    return y / (2 * dilation)


def sfb2d_atrous(
    ll: Tensor,
    lh: Tensor,
    hl: Tensor,
    hh: Tensor,
    filts: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
) -> Tensor:
    """Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to `sfb1d_atrous`
    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    if len(filts) == 2:
        g0, g1 = filts
        g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
    elif len(filts) == 4:
        g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
    else:
        msg = "Unknown form for input filts"
        raise ValueError(msg)

    lo = sfb1d_atrous(ll, lh, g0_col, g1_col, dim=2)
    hi = sfb1d_atrous(hl, hh, g0_col, g1_col, dim=2)
    return sfb1d_atrous(lo, hi, g0_row, g1_row, dim=3)


class SWTForward(nn.Module):
    """Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
    """

    def __init__(self, J: int = 1, wave: str = "db1", mode: str = "periodic") -> None:
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)  # type: ignore[reportAttributeAccessIssue]
        if isinstance(wave, pywt.Wavelet):  # type: ignore[reportAttributeAccessIssue]
            h0_col, h1_col = wave.dec_lo, wave.dec_hi  # type: ignore[reportAttributeAccessIssue]
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 2:
            h0_col, h1_col = wave[0], wave[1]  # type: ignore[reportAssignmentType]
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 4:
            h0_col, h1_col = wave[0], wave[1]  # type: ignore[reportAssignmentType]
            h0_row, h1_row = wave[2], wave[3]  # type: ignore[reportAssignmentType]
        else:
            raise NotImplementedError

        # Prepare the filters
        filts: tuple[Tensor, Tensor, Tensor, Tensor] = prep_filt_afb2d(
            h0_col,  # type: ignore[reportArgumentType]
            h1_col,  # type: ignore[reportArgumentType]
            h0_row,  # type: ignore[reportArgumentType]
            h1_row,  # type: ignore[reportArgumentType]
        )
        with torch.no_grad():
            self.h0_col = nn.Parameter(filts[0])
            self.h1_col = nn.Parameter(filts[1])
            self.h0_row = nn.Parameter(filts[2])
            self.h1_row = nn.Parameter(filts[3])

        self.J = J
        self.mode = mode

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass of the SWT.

        Args:
        ----
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).

        """
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts: tuple[Tensor, Tensor, Tensor, Tensor] = (
            self.h0_col,
            self.h1_col,
            self.h0_row,
            self.h1_row,
        )
        for _j in range(self.J):
            # Do 1 level of the transform
            y = afb2d_atrous(ll, filts)
            coeffs.append(y)

        return coeffs


@torch.no_grad()
def wavelet_guided(output: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    wavelet = pywt.Wavelet("sym7")  # type: ignore[reportAttributeAccessIssue]
    dlo = wavelet.dec_lo
    an_lo = np.divide(dlo, sum(dlo))
    an_hi = wavelet.dec_hi
    rlo = wavelet.rec_lo
    syn_lo = 2 * np.divide(rlo, sum(rlo))
    syn_hi = wavelet.rec_hi

    filters = pywt.Wavelet("wavelet_normalized", [an_lo, an_hi, syn_lo, syn_hi])  # type: ignore[reportAttributeAccessIssue]
    sfm = SWTForward(1, filters, "periodic").to(
        gt.device, dtype=gt.dtype, non_blocking=True
    )

    # wavelet bands of sr image
    sr_img_y = 16.0 + (
        output[:, 0:1, :, :] * 65.481
        + output[:, 1:2, :, :] * 128.553
        + output[:, 2:, :, :] * 24.966
    )
    wavelet_sr: Tensor = sfm(sr_img_y)[0]

    # LL = wavelet_sr[:, 0:1, :, :]
    LH: Tensor = wavelet_sr[:, 1:2, :, :]
    HL: Tensor = wavelet_sr[:, 2:3, :, :]
    HH: Tensor = wavelet_sr[:, 3:, :, :]

    combined_HF = torch.cat((LH, HL, HH), dim=1)

    # wavelet bands of hr image
    hr_img_y = 16.0 + (
        gt[:, 0:1, :, :] * 65.481
        + gt[:, 1:2, :, :] * 128.553
        + gt[:, 2:, :, :] * 24.966
    )
    wavelet_hr: Tensor = sfm(hr_img_y)[0]

    # LL_gt = wavelet_hr[:, 0:1, :, :]
    LH_gt: Tensor = wavelet_hr[:, 1:2, :, :]
    HL_gt: Tensor = wavelet_hr[:, 2:3, :, :]
    HH_gt: Tensor = wavelet_hr[:, 3:, :, :]
    combined_HF_gt = torch.cat((LH_gt, HL_gt, HH_gt), dim=1)

    return combined_HF, combined_HF_gt
