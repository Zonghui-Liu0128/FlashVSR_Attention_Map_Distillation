import torch


def _flatten_video_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Collapse the time dim into the batch dim so LPIPS sees per-frame BCHW.

    4D (B, C, H, W)         -> returned unchanged.
    5D (B, C, T, H, W)      -> reshaped to (B*T, C, H, W) by permuting T next to B.
    Anything else raises -- callers must pass video tensors, not flattened.
    """
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
    raise ValueError(
        f"L_lpips expected 4D or 5D tensor, got ndim={x.ndim} shape={tuple(x.shape)}"
    )


def L_lpips(x_s_latent, gt_hr_rgb, vae_decoder, lpips_net) -> torch.Tensor:
    """Per-frame LPIPS in RGB space: decode(x_s) vs GT_HR, averaged over frames.

    Production: tc_decoder returns (B, 3, T_rgb, H, W); dataset HR is also
    (B, 3, T_rgb, H, W). Both get flattened to (B*T_rgb, 3, H, W) before
    `lpips_net` is invoked, so it sees per-frame BCHW.

    Mixed cases (one input 5D, other 4D) are handled by broadcasting the 4D
    side along the batch axis after flattening -- useful for test stubs and
    sanity-check pipelines where GT is a single still frame.
    """
    rgb_s = _flatten_video_to_bchw(vae_decoder(x_s_latent))
    gt_bchw = _flatten_video_to_bchw(gt_hr_rgb)

    if rgb_s.shape[0] != gt_bchw.shape[0]:
        if gt_bchw.shape[0] == 1:
            gt_bchw = gt_bchw.expand_as(rgb_s).contiguous()
        elif rgb_s.shape[0] == 1:
            rgb_s = rgb_s.expand_as(gt_bchw).contiguous()
        else:
            raise ValueError(
                f"L_lpips batch dim mismatch (after T-flatten): "
                f"rgb_s={tuple(rgb_s.shape)}, gt={tuple(gt_bchw.shape)}"
            )

    return lpips_net(rgb_s, gt_bchw).mean()
