import torch


def L_lpips(x_s_latent, gt_hr_rgb, vae_decoder, lpips_net) -> torch.Tensor:
    """LPIPS in RGB space: decode(x_s) vs GT_HR via lpips_net."""
    rgb_s = vae_decoder(x_s_latent)
    return lpips_net(rgb_s, gt_hr_rgb).mean()
