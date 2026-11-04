"""SigLIP So400m-384 text embeddings for prompt sets (the zero-shot head's W), from open_clip's own tokenizer
and text tower. Only the text tower's tensors are read from the checkpoint (safetensors, lazy), so this needs
about 2 GB of RAM. Run with the project venv:

    python tools/hardcases/text_embed.py deployed generic subtypes

``deployed`` re-derives exports/final/head.npz["W"] from compclass.zeroshot's prompts; it must match before any
other set is trusted. Each set is saved to ART/text/<set>.npz: ``W`` [C, D] (normalised mean over the class's
prompts) and, per class, ``P_<i>`` [n_prompts, D] (each prompt normalised) for max-over-prompts heads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ART, deployed_W, dev

ARCH = "ViT-SO400M-14-SigLIP-384"
REPO_ID = "timm/ViT-SO400M-14-SigLIP-384"

GENERIC_TEMPLATES = [
    "a photo of a {}.", "a close-up photo of a {}.", "a photo of a {}, an electronic component.",
    "a product photo of a {}.", "a {} on a white background.", "a photo of a {} on a circuit board.",
    "a photo of the {}.", "a cropped photo of a {}.", "a bright photo of a {}.", "a photo of a small {}.",
    "a photo of a {} on a table.", "an electronics store photo of a {}.", "a macro photo of a {}.",
    "a photo of a {} used in electronics projects.", "a picture of a {}.", "a photo of several {}s.",
]

SUBTYPE_TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "a product photo of {}."]

# Common physical forms of each class, written for all 20 classes from general component knowledge. Several
# phrases for the five hard pairs (LED legs, DIP switch, SMD power inductor, Arduino board, wired transformer)
# follow the failure types the owner described; they were not written from the demo pictures themselves.
SUBTYPES = {
    "battery holder": ["an AA battery holder", "a coin cell battery holder", "a 9V battery clip",
                       "an 18650 battery holder", "a plastic battery box with wires"],
    "breadboard": ["a solderless breadboard", "a half-size breadboard", "a white breadboard with holes",
                   "a mini breadboard", "an empty breadboard"],
    "capacitor": ["an electrolytic capacitor", "a ceramic disc capacitor", "an SMD capacitor",
                  "a film capacitor", "a tantalum capacitor"],
    "connector": ["a pin header connector", "a screw terminal block", "a USB connector", "a JST connector",
                  "a D-sub connector"],
    "crystal oscillator": ["a quartz crystal oscillator", "an HC-49 crystal in a metal can",
                           "a metal can crystal oscillator", "an SMD crystal oscillator"],
    "diode": ["a rectifier diode with a stripe", "a black 1N4007 diode", "a glass zener diode",
              "an SMD diode", "a Schottky diode"],
    "fuse": ["a glass cartridge fuse", "a blade fuse", "a fuse holder", "a ceramic fuse",
             "a resettable polyfuse"],
    "heat sink": ["an aluminium heat sink", "a finned heat sink", "a CPU heat sink", "a TO-220 heat sink"],
    "ic chip": ["an integrated circuit chip in a DIP package", "an SMD integrated circuit", "a black microchip",
                "a QFP chip with many pins", "an SOIC chip"],
    "inductor coil": ["a toroidal inductor coil", "a copper wire coil", "an SMD power inductor",
                      "a radial inductor", "a ferrite core inductor"],
    "jumper wire": ["jumper wires", "dupont jumper wires", "a bundle of colourful jumper wires",
                    "male-to-male jumper wires"],
    "led": ["an LED with long legs", "a 5mm LED", "an RGB LED", "coloured LEDs", "an SMD LED",
            "a glowing LED"],
    "microcontroller board": ["an Arduino board", "an Arduino Uno", "an ESP32 development board",
                              "a Raspberry Pi Pico", "a microcontroller development board",
                              "an Arduino Nano on a breadboard"],
    "potentiometer": ["a rotary potentiometer", "a trimmer potentiometer", "a slide potentiometer",
                      "a potentiometer with a knob"],
    "power supply module": ["a power supply module", "a DC-DC buck converter module",
                            "a breadboard power supply module", "a switching power supply board"],
    "relay": ["an electromechanical relay", "a relay module", "a 5V relay", "an automotive relay"],
    "resistor": ["a through-hole resistor with colour bands", "an SMD resistor", "a carbon film resistor",
                 "a resistor network"],
    "switch": ["a DIP switch", "a toggle switch", "a push button switch", "a slide switch",
               "a tactile switch", "a rocker switch"],
    "transformer": ["a small transformer", "a transformer with wire leads", "a toroidal transformer",
                    "a power transformer", "an audio transformer", "a PCB mount transformer"],
    "transistor": ["a TO-92 transistor", "a TO-220 power transistor", "a MOSFET", "an SMD transistor"],
}


def prompt_sets(classes: list[str]) -> dict[str, list[list[str]]]:
    from compclass.zeroshot import PROMPT_NAMES, TEMPLATES

    name = lambda c: PROMPT_NAMES.get(c, c)
    assert sorted(SUBTYPES) == classes
    return {
        "deployed": [[t.format(name(c)) for t in TEMPLATES] for c in classes],
        "generic": [[t.format(name(c)) for t in GENERIC_TEMPLATES] for c in classes],
        "subtypes": [[t.format(s) for s in SUBTYPES[c] for t in SUBTYPE_TEMPLATES] for c in classes],
        "subtypes_named": [[t.format(name(c)) for t in TEMPLATES] +
                           [t.format(s) for s in SUBTYPES[c] for t in SUBTYPE_TEMPLATES] for c in classes],
    }


def text_tower(device: torch.device):
    import open_clip
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    with torch.device("meta"):
        model = open_clip.create_model(ARCH, pretrained=None, device="meta")
    text = model.text
    del model
    path = hf_hub_download(REPO_ID, "open_clip_model.safetensors")
    with safe_open(path, "pt") as f:
        sd = {k[len("text."):]: f.get_tensor(k) for k in f.keys() if k.startswith("text.")}  # noqa: SIM118
    text.load_state_dict(sd, strict=True, assign=True)
    left = [n for n, t in [*text.named_parameters(), *text.named_buffers()] if t.is_meta]
    if left:
        raise SystemExit(f"text tower tensors not loaded: {left}")
    return text.eval().to(device), open_clip.get_tokenizer(ARCH)


@torch.no_grad()
def embed(text, tok, prompts: list[str], device: torch.device) -> np.ndarray:
    out = []
    for i in range(0, len(prompts), 32):
        e = text(tok(prompts[i:i + 32]).to(device)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    names = sys.argv[1:]
    device = torch.device("cpu")
    torch.set_num_threads(8)
    classes = dev().classes
    sets = prompt_sets(classes)
    text, tok = text_tower(device)
    (ART / "text").mkdir(parents=True, exist_ok=True)
    for s in names:
        P = [embed(text, tok, ps, device) for ps in sets[s]]
        W = np.stack([p.mean(0) / np.linalg.norm(p.mean(0)) for p in P]).astype(np.float32)
        np.savez(ART / "text" / f"{s}.npz", W=W, **{f"P_{i}": p for i, p in enumerate(P)})
        cos = (W * deployed_W()).sum(1)
        print(f"{s}: {sum(len(p) for p in P)} prompts; cosine to deployed W per class min {cos.min():.6f} "
              f"mean {cos.mean():.6f}", flush=True)
        if s == "deployed" and cos.min() < 0.9999:
            raise SystemExit("text pipeline does not reproduce the deployed W")


if __name__ == "__main__":
    main()
