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
        self.models = {}
        
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

    def set_dataset(self, ctx, dataset, model):
        if not ctx.guild.id in self.models:
            self.models[ctx.guild.id] = {}

        self.models[ctx.guild.id][dataset] = model
        
    
    def load_dataset(self, ctx, dataset):
        if ctx.guild.id in self.models:
            if dataset in self.models[ctx.guild.id]:
                return self.models[ctx.guild.id][dataset]

        with open(self.cache(ctx) + dataset, "r") as f:
            model = IntelliText.from_json(f.read())
            
        self.set_dataset(ctx, dataset, model)

        return model

    def remove_dataset(self, ctx, dataset):
        if ctx.guild.id in self.models:
            if dataset in self.models[ctx.guild.id]:
                del self.models[ctx.guild.id][dataset]
        
    @commands.hybrid_command(name="generate", help="Generate text")
    @discord.app_commands.describe(
        dataset="The name of the dataset to generate from",
        sentences="The number of sentences to generate",
    )
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def generate(self, ctx: commands.Context, dataset: str, sentences: int = 3):
        await ctx.defer()

        if not os.path.exists(self.cache(ctx) + dataset):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return
            
        await ctx.send(
            embed=discord.Embed(
                title=f"Generated from {dataset}",
                description=self.generate_sentences(
                    self.load_dataset(ctx, dataset), max(min(sentences, 10), 1)
                ),
            )
        )

    @commands.hybrid_group(name="dataset", fallback="list", help="List datasets")
    async def dataset(self, ctx: commands.Context):
        embed = discord.Embed(
            title=f"{ctx.guild.name}'s datasets",
            description="\n".join(os.listdir(self.cache(ctx))),
        )

        if ctx.guild.id in self.models:
            if len(self.models[ctx.guild.id]) > 0:
                embed.add_field(name="Cached Models", value=", ".join(self.models[ctx.guild.id].keys()))

        await ctx.send(
            embed=embed,
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
            model = IntelliText(f.read(), well_formed=False)

        self.set_dataset(ctx, name, model)

        with open(self.cache(ctx) + name, "w") as f:
            f.write(model.to_json())

        await ctx.send(f"Successfully added and cached {name}!", ephemeral=True)

    @dataset.command(name="remove", help="Remove a dataset")
    @discord.app_commands.describe(dataset="The name of the dataset to remove")
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def remove_dataset(self, ctx: commands.Context, dataset: str):
        cache_path = self.cache(ctx) + dataset

        if not os.path.exists(cache_path):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return
            
        os.remove(cache_path)

        self.remove_dataset(ctx, dataset)

        await ctx.send(f"Successfully removed {dataset}.", ephemeral=True)

    @dataset.command(name="regenerate", help="Regenerate the cache for a dataset")
    @discord.app_commands.describe(dataset="The name of the dataset to regenerate", token_size="How many words should be in a token. Higher numbers is more coherent but much less random. (default 2)")
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def regenerate_dataset(self, ctx: commands.Context, dataset: str, token_size: int = 2):
        await ctx.defer()
        
        input_path = self.inputs(ctx) + dataset
        cache_path = self.cache(ctx) + dataset

        if not os.path.exists(cache_path):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return

        with open(input_path, "r") as f:
            model = IntelliText(f.read(), well_formed=False, state_size=token_size)

        self.set_dataset(ctx, dataset, model)
        
        with open(cache_path, "w") as f:
            f.write(model.to_json())

        await ctx.send(f"Successfully regenerated {dataset}.", ephemeral=True)

    @dataset.command(name="combine", help="Combine two datasets")
    @discord.app_commands.describe(new_name="What to name the new dataset", dataset_1="The first dataset to include", dataset_2="The second dataset to include")
    @discord.app_commands.autocomplete(dataset_1=autocomplete_dataset, dataset_2=autocomplete_dataset)
    async def regenerate_dataset(self, ctx: commands.Context, new_name: str, dataset_1: str, dataset_2: str):
        await ctx.defer()
        
        for dataset in [dataset_1, dataset_2]:
            if not os.path.exists(self.cache(ctx) + dataset):
                await ctx.send(f"The dataset '{dataset}' does not exist.", ephemeral=True)
                return

        models = [self.load_dataset(ctx, dataset) for dataset in [dataset_1, dataset_2]]
        
        model = markovify.combine(models)

        self.set_dataset(ctx, new_name, model)
        
        with open(self.cache(ctx) + new_name, "w") as f:
            f.write(model.to_json())

        await ctx.send(f"Successfully created {new_name}.", ephemeral=True)

    @dataset.command(name="get", help="Download a dataset")
    @discord.app_commands.describe(dataset="The name of the dataset to download")
    @discord.app_commands.autocomplete(dataset=autocomplete_dataset)
    async def get_dataset(self, ctx: commands.context, dataset: str):
        await ctx.defer()
        
        path = self.inputs(ctx) + dataset

        if not os.path.exists(path):
            await ctx.send("That dataset does not exist.", ephemeral=True)
            return

        await ctx.send(file=discord.File(path), ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    nltk.download("averaged_perceptron_tagger")

    await bot.add_cog(Markov(bot))
