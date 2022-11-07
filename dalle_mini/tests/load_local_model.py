""" Try to load model from local path.

Still struggling on wandb model (dalle-mega).
- https://github.com/wandb/wandb
"""
import jax.numpy as jnp
from flax.jax_utils import replicate
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from dalle_mini import DalleBartProcessor

# dalle-mega
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  
# DALLE_MODEL = "/home/t36668/.cache/wandb/artifacts/obj/md5/"
# can be wandb artifact or HugglingFace Hub or local folder or google bucket
DALLE_COMMIT_ID = None


# VQGAN model
# VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_REPO = "/home/t36668/.cache/huggingface/hub/models--dalle-mini--vqgan_imagenet_f16_16384/snapshots/e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

params = replicate(params)
vqgan_params = replicate(vqgan_params)

