# build_results subpackage
# Note: BuildResults imported on demand to avoid circular import when running as __main__
__all__ = ['BuildResults']

def __getattr__(name):
    if name == 'BuildResults':
        from .run_on_systems import BuildResults
        return BuildResults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")





