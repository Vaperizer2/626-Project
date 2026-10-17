# OpenRGB SSH status presets

Use these commands on the compute rig (`ai-pc-04`) to show the agent state on the `B860M D3HP` motherboard RGB controller.

The `--noautoconnect` flag is intentional: it skips the optional OpenRGB SDK server connection and goes straight to hardware detection. The known-good Training profile uses controller-wide `Direct` mode with 32 per-LED colors; the commands below mirror that profile instead of using `Static` mode per zone.

## Quick copy-paste commands

Run this helper once after SSHing into the rig:

```bash
set_b860m_rgb() {
    local color="${1:?color required, e.g. FFFFFF}"
    local brightness="${2:-100}"
    sudo openrgb --noautoconnect \
        --device "B860M D3HP" \
        --mode direct \
        --color "$color" \
        --brightness "$brightness"
}
```

Then use one of these presets:

```bash
# BUSY / actively training — solid white
set_b860m_rgb FFFFFF 100

# RUNNING / active but non-critical — blue
set_b860m_rgb 007BFF 100

# SUCCESS / completed — green
set_b860m_rgb 00FF66 100

# WAITING / queued — amber
set_b860m_rgb FFA500 100

# BLOCKED / failed or needs attention — red
set_b860m_rgb FF0000 100

# IDLE / available — lights off
set_b860m_rgb 000000 0
```

## One-command SSH versions

Run these from another machine. `-t` gives `sudo` a terminal if it needs the password.

```bash
# Busy / white (profile-equivalent Direct mode)
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color FFFFFF --brightness 100'

# Running / blue
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color 007BFF --brightness 100'

# Success / green
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color 00FF66 --brightness 100'

# Waiting / amber
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color FFA500 --brightness 100'

# Blocked / red
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color FF0000 --brightness 100'

# Idle / off
ssh -t ai-pc-04 'sudo openrgb --noautoconnect --device "B860M D3HP" --mode direct --color 000000 --brightness 0'
```

## Load the saved white profile

The Training user has a valid saved profile at `/home/carlai04/.config/OpenRGB/Default.orp`. It contains white LED values for this motherboard and is the preferred way to restore the known-good white state:

```bash
sudo openrgb --noautoconnect --profile /home/carlai04/.config/OpenRGB/Default.orp
```

The current OpenRGB configuration does not enable profile autostart, so this command must be run explicitly unless autostart is enabled later.

## Optional Bash helper

Paste this once into a Bash shell, then use `rgb_status busy`, `rgb_status success`, and so on:

```bash
rgb_status() {
    local status="${1:-}"
    local color brightness

    case "$status" in
        busy)    color=FFFFFF; brightness=100 ;; # actively training
        running) color=007BFF; brightness=100 ;; # active work
        success) color=00FF66; brightness=100 ;; # completed successfully
        waiting) color=FFA500; brightness=100 ;; # queued / waiting
        blocked) color=FF0000; brightness=100 ;; # failed / needs attention
        idle)    color=000000; brightness=0   ;; # available
        *)
            echo "Usage: rgb_status {busy|running|success|waiting|blocked|idle}" >&2
            return 2
            ;;
    esac

    sudo openrgb --noautoconnect \
        --device "B860M D3HP" \
        --mode direct \
        --color "$color" \
        --brightness "$brightness"
}
```

## Recovery / verification

If the motherboard name changes or OpenRGB reports an empty device ID, load I²C support and list devices:

```bash
sudo modprobe i2c-dev
sudo openrgb --noautoconnect --list-devices --verbose
```

The expected controller is currently:

```text
0: B860M D3HP
Type: Motherboard
Description: IT82950-LED V0.0.0.5
Zones: ARGB_V2_1 ARGB_V2_2
```

The training job and its datasets are independent of these lighting commands.
