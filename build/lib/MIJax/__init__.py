from importlib.metadata import version

try:

    __version__ = version('MIJax')

except ImportError:

    pass
    

