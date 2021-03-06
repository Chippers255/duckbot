from discord.ext import commands
import subprocess


class Fortune(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="fortune")
    async def fortune(self, context):
        await context.send(self.get_fortune())

    def get_fortune(self):
        message = self.__get_fortune_output()
        while len(message) > 1950:  # discord has a 2000 character limit, just leaving a buffer
            message = self.__get_fortune_output()
        return f"```{message}```"

    def __get_fortune_output(self):
        process = subprocess.run("fortune -a | cowsay -f $(ls /usr/share/cowsay/cows/ | shuf -n1)", shell=True, capture_output=True)
        return process.stdout.decode()
