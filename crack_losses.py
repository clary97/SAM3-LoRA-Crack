"""
clDice (centerline Dice) loss for thin crack structures, plugged into SAM3's
mask loss without modifying the vendored sam3 package.

clDice (Shit et al., CVPR 2021) is a topology-aware loss: it matches the
*skeletons* of prediction and ground truth, so connectivity of thin structures
is preserved — exactly the weak spot for crack segmentation, where ordinary
Dice/IoU is dominated by a 1-2px boundary error.

`MasksClDice` subclasses sam3's `Masks` and adds a `loss_cldice` term on the
2D matched masks (only valid on the non-point-sampled path, which this project
uses). Enable it by adding `loss_cldice` to the `loss:` section of the config.
"""
import torch
import torch.nn.functional as F

from sam3.train.loss.loss_fns import Masks, dice_loss, sigmoid_focal_loss


# ---- differentiable soft skeleton (iterative morphological thinning) ----
def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skel(x, iters=10):
    """Soft skeleton of x in [0,1], shape [N,1,H,W]."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iters=10, smooth=1.0):
    """clDice loss. pred,target: [N,1,H,W]; pred in [0,1], target in {0,1}.

    Returns 1 - clDice, averaged over the batch.
    """
    sp = soft_skel(pred, iters)
    st = soft_skel(target, iters)
    dims = (1, 2, 3)
    tprec = (sp * target).sum(dims).add(smooth) / (sp.sum(dims) + smooth)   # skel(pred) inside target
    tsens = (st * pred).sum(dims).add(smooth) / (st.sum(dims) + smooth)     # skel(target) inside pred
    cldice = 2.0 * tprec * tsens / (tprec + tsens)
    return (1.0 - cldice).mean()


class MasksClDice(Masks):
    """Masks loss + clDice. Adds key 'loss_cldice' (put it in weight_dict)."""

    def __init__(self, *args, cldice_iters=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldice_iters = cldice_iters

    def get_loss(self, outputs, targets, indices, num_boxes):
        assert self.num_sample_points is None, \
            "MasksClDice only supports the non-point-sampled mask path"
        assert "pred_masks" in outputs and "is_valid_mask" in targets

        src_masks = outputs["pred_masks"]
        if targets["masks"] is None:
            z = torch.tensor(0.0, device=src_masks.device)
            return {"loss_mask": z, "loss_dice": z, "loss_cldice": z}

        target_masks = targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        target_masks = target_masks.to(src_masks)
        keep = targets["is_valid_mask"] if indices[2] is None else targets["is_valid_mask"][indices[2]]
        src_masks = src_masks[(indices[0], indices[1])][keep]
        target_masks = target_masks[keep]

        if target_masks.shape[0] == 0 and src_masks.shape[0] == 0:
            src_flat = src_masks.flatten(1)
            tgt_flat = target_masks.reshape(src_flat.shape)
            return {
                "loss_mask": sigmoid_focal_loss(src_flat, tgt_flat, num_boxes,
                                                alpha=self.focal_alpha, gamma=self.focal_gamma),
                "loss_dice": dice_loss(src_flat, tgt_flat, num_boxes),
                "loss_cldice": torch.tensor(0.0, device=src_masks.device),
            }

        if len(src_masks.shape) == 3:
            src_masks = src_masks[:, None]
        if src_masks.dtype == torch.bfloat16:
            src_masks = src_masks.to(dtype=torch.float32)
        src_masks = F.interpolate(src_masks, size=target_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        src_2d = src_masks[:, 0]                       # [N,H,W] logits
        loss_cldice = soft_cldice_loss(
            src_2d.sigmoid()[:, None], target_masks[:, None].float(), self.cldice_iters)

        src_flat = src_2d.flatten(1)
        tgt_flat = target_masks.flatten(1)
        return {
            "loss_mask": sigmoid_focal_loss(src_flat, tgt_flat, num_boxes,
                                            alpha=self.focal_alpha, gamma=self.focal_gamma),
            "loss_dice": dice_loss(src_flat, tgt_flat, num_boxes),
            "loss_cldice": loss_cldice,
        }
