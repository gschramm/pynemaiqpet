from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pynemaiqpet")
except PackageNotFoundError:
    __version__ = "unknown"
