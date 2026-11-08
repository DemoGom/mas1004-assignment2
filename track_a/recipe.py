"""Your Track A recipe: how the fixed model is trained on your selection.

src/official.py imports this file and calls the functions below. Change
anything in it. The grader copies this one file and your selection.csv to its
own computer and runs src/official.py with them, so everything your recipe
needs has to be in this file: it cannot read other files, download anything,
or load another model.

What you cannot change, because official.py decides it:
  - the model: ResNet18 with the ImageNet weights, a new last layer of 5
  - the photos: only the ones in selection.csv, with the city of their folder
  - how many photos go through the network in training: rules.TRACK_A_BUDGET
    (250,000). A bigger batch means fewer steps, not more photos.
  - the image size the model sees, 224 x 224
  - mixed precision (float16), and the seed

The starter recipe, with one change: the model that is handed in is a moving
average of the weights over the whole run (on_step, final_model), not the
weights of the very last step. Everything else is the starter recipe.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms as T

BATCH_SIZE = 64


def train_transform():
    """A function that turns one training photo (a PIL image) into a 224 x 224 one.

    It may return a PIL image or a float tensor (3, 224, 224) with values
    0..1. official.py normalises it afterwards. It runs once every time a
    photo is drawn, so the same photo looks different every time.
    """
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
    ])


def sample_weights(cities, paths):
    """How often each photo in your selection is drawn.

    cities and paths are lists, one entry per row of selection.csv. Return a
    list of positive numbers of the same length, where a photo with weight 2
    is drawn twice as often as one with weight 1. Return None to go through
    the selection in a new random order each time, every photo once per round.
    """
    return None


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)


def make_scheduler(optimizer, total_steps):
    """Stepped once after every training step. total_steps is BUDGET // BATCH_SIZE."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.15)


def mix(images, labels):
    """Change a batch on the GPU before the model sees it.

    images is (BATCH_SIZE, 3, 224, 224), normalised; labels is (BATCH_SIZE,)
    city numbers. Return images of the same shape and the targets for
    loss_fn: the labels, or probabilities (BATCH_SIZE, 5) if you mix photos.
    """
    return images, labels


def loss_fn(logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=0.1)


_ema = None


def on_step(step, total_steps, model, optimizer):
    """Called after every step, with the step number from 0. Returns nothing.

    Here it keeps a moving average of the weights: EMA = 0.999 * EMA + 0.001 *
    weights. With 3,906 steps it averages roughly the last thousand steps. The
    learning rate falls to almost zero at the end of the one-cycle schedule, so
    the last weights are already quiet; averaging over the end of training
    removes what is left of the noise.
    """
    global _ema
    decay = 0.999
    with torch.no_grad():
        if _ema is None:
            _ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    _ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
                else:
                    _ema[k].copy_(v)


def final_model(model):
    """The model to hand in, at the end of training. It must still be the same ResNet18.

    It is the moving average of the weights, not the last step.
    """
    if _ema is not None:
        model.load_state_dict(_ema)
    return model
