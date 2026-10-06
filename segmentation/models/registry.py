def register_model(fn=None, **kwargs):
    """Minimal no-op decorator so the backbone's register_model import resolves."""
    def deco(f):
        return f
    return deco if fn is None else fn
