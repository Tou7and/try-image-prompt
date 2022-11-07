""" Inference pipeline from dalle-mini

source:
    https://github.com/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb

2022.11.07, JamesH.
"""
import os
import random
import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
# from transformers import CLIPProcessor, FlaxCLIPModel
from functools import partial
from flax.training.common_utils import shard_prng_key
from PIL import Image

DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
# DALLE_MODEL = "/home/t36668/.cache/huggingface/hub/models--dalle-mini--dalle-mini"
DALLE_COMMIT_ID = None
# VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_REPO = "/home/t36668/.cache/huggingface/hub/models--dalle-mini--vqgan_imagenet_f16_16384/snapshots/e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

print("Loading dalle-mini ...")
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

params = replicate(params)
vqgan_params = replicate(vqgan_params)

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# create a random key
SEED = random.randint(0, 2**32 - 1)
KEY = jax.random.PRNGKey(SEED)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

def get_image(prompt_string="Sonic caught by the camera", output_dir="exp/img"):
    """ Given prompt, generate images. """
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    prompts = [prompt_string]
    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)

    # number of predictions per prompt
    n_predictions = 1

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    print(f"Prompts: {prompts}\n")
    # generate images
    images = []
    # for i in trange(max(n_predictions // jax.device_count(), 1)):
    # get a new key
    key, subkey = jax.random.split(KEY)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for idx, decoded_img in enumerate(decoded_images):
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)
        img.save(os.path.join(output_dir, "{}.png".format(idx)), "PNG")

if __name__ == "__main__":
    get_image()
