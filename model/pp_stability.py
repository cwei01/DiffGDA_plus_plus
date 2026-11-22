"""Optional source-only training stabilizers; neutral defaults preserve PP."""
import torch


def source_class_weights(labels,classes,power=0.):
    if power<=0:return None
    counts=torch.bincount(labels,minlength=classes).to(dtype=torch.float32)
    if (counts==0).any():raise ValueError('Class balancing requires source coverage')
    weights=counts.pow(-power)
    # Mean weight over source nodes is one; no target class information enters.
    weights/=((weights*counts).sum()/counts.sum())
    return weights.to(labels.device)


@torch.no_grad()
def update_ema(average,model,decay):
    original=model.state_dict()
    for name,value in average.state_dict().items():
        source=original[name]
        if value.is_floating_point():value.mul_(decay).add_(source,alpha=1-decay)
        else:value.copy_(source)
