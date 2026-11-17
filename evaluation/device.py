"""Map the policy CUDA device to the same physical renderer device."""
import re
import subprocess


def pci_render_device(uuid):
    identifier = str(uuid)
    if not identifier.startswith(("GPU-", "MIG-")):
        identifier = "GPU-" + identifier
    bus = subprocess.check_output(
        ["nvidia-smi", "-i", identifier, "--query-gpu=pci.bus_id",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    if not re.fullmatch(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", bus):
        raise ValueError("Cannot identify the renderer PCI device for the policy GPU.")
    return "pci:" + bus.split(":", 1)[1]
