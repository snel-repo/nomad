import functools

def rgetattr(obj, attr, *args):
    """ a recursive drop-in replacement for getattr, which also handles dotted attr strings """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))