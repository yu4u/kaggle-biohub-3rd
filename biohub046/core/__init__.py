# MONAI must be imported before the cv2/albumentations + pytorch_lightning stack,
# otherwise building a MONAI net later segfaults (SIGSEGV during SegResNet
# construction; reproduced deterministically with `import src.datamodule` first,
# and fixed by importing monai first). Doing it here means any `import src.*`
# gets the safe order for free. Guarded so environments without monai (only
# arch=segresnet needs it) keep working.
try:  # noqa: SIM105
    import monai  # noqa: F401
except ImportError:
    pass
