import torch


def blacken_sky(image: torch.Tensor, sky_mask, enabled: bool) -> torch.Tensor:
    if not enabled or sky_mask is None:
        return image

    mask = sky_mask.to(device=image.device).bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and image.shape[0] != 1:
        mask = mask.expand(image.shape[0], -1, -1)
    return torch.where(mask, torch.zeros_like(image), image)


def has_mask_pixels(mask) -> bool:
    return mask is not None and bool(torch.count_nonzero(mask).item() > 0)
