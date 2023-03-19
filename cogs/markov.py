import discord  # type: ignore
from discord.ext import commands  # type: ignore
from utils.funcs import *
from utils.jdb import JSONDatabase as jdb
import markovify
import os
import nltk
import re
from typing import List


class IntelliText(markovify.Text):
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = ["::".join(tag) for tag in nltk.pos_tag(words)]
        return words

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence


class Markov(commands.Cog):
    """Markov Chains"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    def get_server_path(self, ctx: commands.Context):
        path = os.path.abspath("./data/" + str(ctx.guild.id) + "/")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path + "/"

    def cache(self, ctx: commands.Context):
        path = self.get_server_path(ctx) + "cache/"

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path

    def inputs(self, ctx: commands.Context):
        path = self.get_server_path(ctx) + "inputs/"

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path

    async def autocomplete_dataset(
        self, interaction: discord.Interaction, current: str
    ) -> List[discord.app_commands.Choice[str]]:
        return [
            discord.app_commands.Choice(name=filename, value=filename)
            for filename in os.listdir(self.cache(interaction))
            if current.lower() in filename.lower()
        ]

    def generate_sentence(self, model) -> str:
        sentence = ""

        while not sentence:
            sentence = model.make_sentence()

        return sentence

    def generate_sentences(self, model, sentence_count: int) -> str:
        sentences = ""

        for _ in range(max(sentence_count, 1)):
            sentences += self.generate_sentence(model) + " "

        return sentences

    @commands.hybrid_command(name="generate", help="Generate text")
    @discord.app_commands.describe(
        dataset="The name of the dataset to generate from",
        sentences="The number of sentences to generate",
    )
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def generate(self, ctx: commands.Context, dataset: str, sentences: int = 3):
        await ctx.defer()

        path = self.cache(ctx) + dataset

        if not os.path.exists(path):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return

        with open(path, "r") as f:
            await ctx.send(
                embed=discord.Embed(
                    title=f"Generated from {dataset}",
                    description=self.generate_sentences(
                        IntelliText.from_json(f.read()), max(min(sentences, 10), 1)
                    ),
                )
            )

    @commands.hybrid_group(name="dataset", fallback="list", help="List datasets")
    async def dataset(self, ctx: commands.Context):
        await ctx.send(
            embed=discord.Embed(
                title=f"{ctx.guild.name}'s datasets",
                description="\n".join(os.listdir(self.inputs(ctx))),
            ),
            ephemeral=True,
        )

    @dataset.command(name="add", help="Add a dataset")
    @discord.app_commands.describe(
        name="The name of the dataset", attachment="The dataset's content"
    )
    async def add_dataset(
        self, ctx: commands.Context, name: str, attachment: discord.Attachment
    ):
        await ctx.defer()

        await attachment.save(self.inputs(ctx) + name)

        with open(self.inputs(ctx) + name, "r") as f:
            with open(self.cache(ctx) + name, "w") as f2:
                f2.write(IntelliText(f.read(), well_formed=False).to_json())

        await ctx.send(f"Successfully added and cached {name}!", ephemeral=True)

    @dataset.command(name="remove", help="Remove a dataset")
    @discord.app_commands.describe(dataset="The name of the dataset to remove")
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def remove_dataset(self, ctx: commands.Context, dataset: str):
        cache_path = self.cache(ctx) + dataset

    @dataset.command(name="get", help="Download a dataset")
    @discord.app_commands.describe(dataset="The name of the dataset to download")
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def get_dataset(self, ctx: commands.context, dataset: str):
        path = self.inputs(ctx) + dataset

        if not os.path.exists(path):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return

        await ctx.send(file=discord.File(path), ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    nltk.download("averaged_perceptron_tagger")

    await bot.add_cog(Markov(bot))
