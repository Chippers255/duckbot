import pytest
from unittest import mock
from duckbot.util.emojis import regional_indicator
from duckbot.cogs.duck import Duck


@pytest.mark.asyncio
@mock.patch("random.random", return_value=0.0001)
async def test_react_duck_random_passes(random, bot, message):
    clazz = Duck(bot)
    await clazz.react_duck(message)
    message.add_reaction.assert_not_called()


@pytest.mark.asyncio
@mock.patch("random.random", return_value=0.000009)
async def test_react_duck_random_fails(random, bot, message):
    clazz = Duck(bot)
    await clazz.react_duck(message)
    message.add_reaction.assert_called_once_with("\U0001F986")


@pytest.mark.asyncio
@mock.patch("random.random", return_value=0)
async def test_react_with_duckbot_not_bot_author(random, bot, message):
    clazz = Duck(bot)
    await clazz.react_with_duckbot(message)
    message.add_reaction.assert_not_called()


@pytest.mark.asyncio
@mock.patch("random.random", return_value=0.01)
async def test_react_with_duckbot_random_fails(random, bot, message):
    message.author = bot.user
    clazz = Duck(bot)
    await clazz.react_with_duckbot(message)
    message.add_reaction.assert_not_called()


@pytest.mark.asyncio
@mock.patch("random.random", return_value=0.009)
async def test_react_with_duckbot_random_passes(random, bot, message):
    message.author = bot.user
    clazz = Duck(bot)
    await clazz.react_with_duckbot(message)
    calls = [
        mock.call(regional_indicator("i")),
        mock.call(regional_indicator("a")),
        mock.call(regional_indicator("m")),
        mock.call(regional_indicator("d")),
        mock.call(regional_indicator("u")),
        mock.call(regional_indicator("c")),
        mock.call(regional_indicator("k")),
        mock.call(regional_indicator("b")),
        mock.call(regional_indicator("o")),
        mock.call(regional_indicator("t")),
    ]
    message.add_reaction.assert_has_calls(calls, any_order=False)


@pytest.mark.asyncio
async def test_github(bot, context):
    clazz = Duck(bot)
    await clazz._Duck__github(context)
    context.send.assert_called_once_with("https://github.com/Chippers255/duckbot")


@pytest.mark.asyncio
async def test_wiki(bot, context):
    clazz = Duck(bot)
    await clazz._Duck__wiki(context)
    context.send.assert_called_once_with("https://github.com/Chippers255/duckbot/wiki")
