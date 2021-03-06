class MockAsyncIterator:
    """A simple mock implementation for `channel.history`
    and other AsyncIterator values."""

    # note, impl is minimal, needs expanding for fit other use cases
    def __init__(self, message):
        self.message = message

    async def flatten(self):
        if self.message:
            return [self.message]
        else:
            return []
