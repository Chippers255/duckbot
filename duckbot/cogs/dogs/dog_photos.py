import io
import json
import urllib

import discord
import numpy
import PIL.Image
import tensorflow
from discord.ext import commands


def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tensorflow.cast(img, tensorflow.uint8)


class DeepDream(tensorflow.Module):
    def __init__(self):
        base_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        # Maximize the activations of these layers
        names = ["mixed3", "mixed5"]
        layers = [base_model.get_layer(name).output for name in names]

        # Create the feature extraction model
        self.model = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)

    def calc_loss(self, img):
        img_batch = tensorflow.expand_dims(img, axis=0)
        layer_activations = self.model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
            loss = tensorflow.math.reduce_mean(act)
            losses.append(loss)

        return tensorflow.reduce_sum(losses)

    @tensorflow.function(
        input_signature=(
            tensorflow.TensorSpec(shape=[None, None, 3], dtype=tensorflow.float32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=[], dtype=tensorflow.float32),
        )
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tensorflow.constant(0.0)
        for n in tensorflow.range(steps):
            with tensorflow.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = self.calc_loss(img)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tensorflow.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tensorflow.clip_by_value(img, -1, 1)

        return loss, img


class DogPhotos(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="dog")
    async def dog_command(self, context):
        await context.send(self.get_dog_image())

    @commands.command(name="dogdream")
    @commands.cooldown(1, 1000)
    async def dream_command(self, context):
        deepdream = DeepDream()

        url, img = self.download_dog_image()
        await context.send(url)
        img = tensorflow.keras.applications.inception_v3.preprocess_input(img)
        img = tensorflow.convert_to_tensor(img)
        step_size = tensorflow.convert_to_tensor(0.025)
        steps_remaining = 100
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = tensorflow.constant(100)
            else:
                run_steps = tensorflow.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, img = deepdream(img, run_steps, tensorflow.constant(step_size))

            result = deprocess(img)
            result = PIL.Image.fromarray(numpy.array(result))
            result.save("dreamy_dog.png", format="PNG")
            await context.send(file=discord.File("dreamy_dog.png"))
        self.dream_command.reset_cooldown(context)

    def get_dog_image(self):
        req = urllib.request.Request("https://dog.ceo/api/breeds/image/random")
        req.add_header("Cookie", "euConsent=true")
        result = json.loads(urllib.request.urlopen(req).read())
        if result.get("status", "ded") != "success" or not result.get("message", None):
            raise RuntimeError("could not fetch a puppy")
        else:
            return result.get("message")

    def download_dog_image(self):
        url = self.get_dog_image()
        name = url.split("/")[-1]
        image_path = tensorflow.keras.utils.get_file(name, origin=url)
        img = PIL.Image.open(image_path)
        img.thumbnail((250, 250))
        return url, numpy.array(img)
