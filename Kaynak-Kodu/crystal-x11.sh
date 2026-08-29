#!/bin/bash
set -eo pipefail

[ ! -e /dev/fd ] && ln -s /proc/self/fd /dev/fd

# Programcı: Rıza Kadir ATALAY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/crystal-build"
CHROOT_DIR="${BUILD_DIR}/chroot"
ISO_DIR="${BUILD_DIR}/iso"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${CHROOT_DIR}"

debootstrap --arch=amd64 --include=locales trixie "${CHROOT_DIR}" http://deb.debian.org/debian

mount --bind /dev "${CHROOT_DIR}/dev"
mount --bind /dev/pts "${CHROOT_DIR}/dev/pts"
mount --bind /proc "${CHROOT_DIR}/proc"
mount --bind /sys "${CHROOT_DIR}/sys"

cat > "${CHROOT_DIR}/etc/apt/sources.list" <<EOF
deb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware
deb-src http://deb.debian.org/debian trixie main contrib
deb http://security.debian.org/debian-security trixie-security main contrib
deb http://deb.debian.org/debian trixie-updates main contrib
EOF

cat > "${CHROOT_DIR}/usr/sbin/policy-rc.d" <<EOF
#!/bin/sh
exit 101
EOF
chmod +x "${CHROOT_DIR}/usr/sbin/policy-rc.d"

mkdir -p "${CHROOT_DIR}/tmp"
chmod 1777 "${CHROOT_DIR}/tmp"

chroot "${CHROOT_DIR}" apt-get update

echo -e "en_US.UTF-8 UTF-8\ntr_TR.UTF-8 UTF-8" >> "${CHROOT_DIR}/etc/locale.gen"
chroot "${CHROOT_DIR}" locale-gen

echo "Europe/Istanbul" > "${CHROOT_DIR}/etc/timezone"
ln -sf /usr/share/zoneinfo/Europe/Istanbul "${CHROOT_DIR}/etc/localtime"
printf "0.0 0 0.0\n0\nUTC\n" > "${CHROOT_DIR}/etc/adjtime"

cat > "${CHROOT_DIR}/etc/default/locale" <<EOF
LANG=tr_TR.UTF-8
LC_ALL=tr_TR.UTF-8
LC_MESSAGES=tr_TR.UTF-8
LANGUAGE=tr_TR:tr
EOF

chroot_install() {
    DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        --no-install-recommends "$@"
}

chroot_install \
    linux-image-amd64 linux-headers-amd64 build-essential dkms initramfs-tools \
    squashfs-tools live-boot live-boot-initramfs-tools live-config live-config-systemd \
    xorriso systemd-sysv network-manager nm-tray wireless-tools wpasupplicant dbus dbus-x11 \
    sudo keyboard-configuration console-setup htop fastfetch vlc vlc-l10n debootstrap cmatrix \
    gdebi qbittorrent flameshot qemu-utils libguestfs-tools guestfs-tools fuse3 gvfs-fuse kpartx \
    fdisk util-linux soundconverter scrot xdotool fonts-noto-color-emoji font-manager

chroot_install \
    task-lxqt-desktop lxqt-core lxqt-config lxqt-admin lxqt-qtplugin lxqt-powermanagement \
    lxqt-notificationd lxqt-policykit pcmanfm-qt pcmanfm-qt-l10n libfm-qt-l10n qterminal openbox \
    obconf-qt git curl

chroot_install \
    xserver-xorg xserver-xorg-core xserver-xorg-input-all xserver-xorg-video-all xinit \
    x11-xserver-utils mesa-utils

chroot_install \
    gparted parted eject e2fsprogs dosfstools ntfs-3g exfatprogs xfsprogs rsync \
    grub-efi-amd64-bin grub-common mtools efibootmgr yad papirus-icon-theme fonts-dejavu \
    fonts-liberation fonts-font-awesome gvfs gvfs-backends udisks2 upower polkitd pkexec \
    numix-gtk-theme qt5ct qt6ct

chroot_install \
    pipewire pipewire-audio pipewire-pulse pipewire-alsa wireplumber libspa-0.2-bluetooth \
    alsa-utils alsa-ucm-conf pavucontrol pavucontrol-qt pulseaudio-utils

mkdir -p "${CHROOT_DIR}/etc/ssl/certs"
cp /etc/ssl/certs/ca-certificates.crt "${CHROOT_DIR}/etc/ssl/certs/ca-certificates.crt"

chroot "${CHROOT_DIR}" bash -c '
    curl -L "https://github.com/rustdesk/rustdesk/releases/download/1.4.9/rustdesk-1.4.9-x86_64.deb" \
        -o /tmp/rustdesk.deb
    apt-get install -fy /tmp/rustdesk.deb
    rm -f /tmp/rustdesk.deb
'

chroot "${CHROOT_DIR}" bash -c '
if [ ! -d /usr/share/themes/Numix ] || ! ls /usr/share/themes/Numix/*openbox* >/dev/null 2>&1; then
    apt-get install -y git curl tar >/dev/null 2>&1 || true
    cd /tmp
    rm -rf numix-gtk-theme
    git ls-remote numixproject/numix-gtk-theme.git >/dev/null 2>&1 && git clone --depth=1 https://github.com/numixproject/numix-gtk-theme.git || true
    if [ -d numix-gtk-theme/openbox-3 ]; then
        mkdir -p /usr/share/themes/Numix
        cp -a numix-gtk-theme/openbox-3 /usr/share/themes/Numix/openbox-3
    elif [ -d numix-gtk-theme/Openbox-3 ]; then
        mkdir -p /usr/share/themes/Numix
        cp -a numix-gtk-theme/Openbox-3 /usr/share/themes/Numix/openbox-3
    else
        for d in numix-gtk-theme/*openbox*; do
            [ -d "$d" ] && mkdir -p /usr/share/themes/Numix && cp -a "$d" /usr/share/themes/Numix/
        done
    fi
fi
'

chroot_install firmware-linux firmware-linux-free firmware-linux-nonfree firmware-misc-nonfree

chroot_install nvidia-kernel-dkms xserver-xorg-video-nvidia libglx-nvidia0 libegl-nvidia0 nvidia-smi

chroot_install \
    lxqt-about-l10n lxqt-config-l10n lxqt-session-l10n lxqt-panel-l10n lxqt-policykit-l10n \
    liblxqt-l10n librsvg2-common libqt5svg5

chroot_install python3 python3-pip python3-venv falkon || true

chroot "${CHROOT_DIR}" pip3 install --break-system-packages --root-user-action=ignore \
    yfinance mplfinance scikit-learn xgboost PyQt6 sentencepiece requests ijson pandas pyarrow

chroot "${CHROOT_DIR}" pip3 install --break-system-packages --root-user-action=ignore \
    --index-url https://download.pytorch.org/whl/cpu torch

mkdir -p "${CHROOT_DIR}/etc/pipewire/pipewire.conf.d"
cat > "${CHROOT_DIR}/etc/pipewire/pipewire.conf.d/10-crystal.conf" <<EOF
context.properties = {
    log.level = 2
}
EOF

mkdir -p "${CHROOT_DIR}/etc/wireplumber/wireplumber.conf.d"
cat > "${CHROOT_DIR}/etc/wireplumber/wireplumber.conf.d/51-crystal.conf" <<EOF
wireplumber.profiles = {
    main = {
        monitor.alsa = required
    }
}
EOF

for svc in sddm lightdm gdm; do
    chroot "${CHROOT_DIR}" systemctl disable $svc 2>/dev/null || true
    chroot "${CHROOT_DIR}" systemctl mask $svc 2>/dev/null || true
done

chroot "${CHROOT_DIR}" apt-get clean
rm -f "${CHROOT_DIR}/usr/sbin/policy-rc.d"

echo "crystal" > "${CHROOT_DIR}/etc/hostname"
cat > "${CHROOT_DIR}/etc/hosts" <<EOF
127.0.0.1 localhost
127.0.1.1 crystal
::1 localhost ip6-localhost ip6-loopback
EOF

chroot "${CHROOT_DIR}" passwd -d root

mkdir -p "${CHROOT_DIR}/etc/systemd/system/getty@tty1.service.d"
cat > "${CHROOT_DIR}/etc/systemd/system/getty@tty1.service.d/override.conf" <<EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear %I \$TERM
Type=idle
EOF

cat > "${CHROOT_DIR}/etc/live/config.conf" <<EOF
LIVE_HOSTNAME="crystal"
LIVE_USERNAME="root"
LIVE_USER_FULLNAME="root"
LIVE_NOROOT=
LIVE_NOAUTOLOGIN=
LIVE_TIMEZONE="Europe/Istanbul"
EOF

rm -f "${CHROOT_DIR}/lib/live/config/1160-openssh-server" "${CHROOT_DIR}/lib/live/config/1170-user-setup" 2>/dev/null || true

chroot "${CHROOT_DIR}" systemctl enable getty@tty1

mkdir -p "${CHROOT_DIR}/etc/systemd/user" "${CHROOT_DIR}/etc/systemd/user/default.target.wants"

for service in pipewire pipewire-pulse wireplumber; do
    case $service in
        pipewire)
            cat > "${CHROOT_DIR}/etc/systemd/user/pipewire.service" <<EOF
[Unit]
Description=PipeWire Multimedia Service
After=pipewire.socket

[Service]
ExecStart=/usr/bin/pipewire
Restart=on-failure

[Install]
WantedBy=default.target
EOF
            ;;
        pipewire-pulse)
            cat > "${CHROOT_DIR}/etc/systemd/user/pipewire-pulse.service" <<EOF
[Unit]
Description=PipeWire PulseAudio
Requires=pipewire.service
After=pipewire.service

[Service]
ExecStart=/usr/bin/pipewire-pulse
Restart=on-failure

[Install]
WantedBy=default.target
EOF
            ;;
        wireplumber)
            cat > "${CHROOT_DIR}/etc/systemd/user/wireplumber.service" <<EOF
[Unit]
Description=Multimedia Service Session Manager
Requires=pipewire.service
After=pipewire.service

[Service]
ExecStart=/usr/bin/wireplumber
Restart=on-failure

[Install]
WantedBy=default.target
EOF
            ;;
    esac
    ln -sf "/etc/systemd/user/${service}.service" "${CHROOT_DIR}/etc/systemd/user/default.target.wants/${service}.service"
done

mkdir -p "${CHROOT_DIR}/run/user/0"
chmod 700 "${CHROOT_DIR}/run/user/0"
chown root:root "${CHROOT_DIR}/run/user/0"

mkdir -p "${CHROOT_DIR}/etc/security/limits.d"
cat > "${CHROOT_DIR}/etc/security/limits.d/audio.conf" <<EOF
root    soft    rtprio    95
root    soft    memlock   unlimited
root    hard    rtprio    95
root    hard    memlock   unlimited
EOF

chroot "${CHROOT_DIR}" systemctl enable NetworkManager

mkdir -p "${CHROOT_DIR}/etc/initramfs-tools/conf.d" "${CHROOT_DIR}/etc/modprobe.d"

cat > "${CHROOT_DIR}/etc/initramfs-tools/initramfs.conf" <<EOF
MODULES=most
BUSYBOX=y
COMPRESS=gzip
BOOT=live
EOF

cat > "${CHROOT_DIR}/etc/initramfs-tools/conf.d/live.conf" <<EOF
export LIVE=true
export BOOT=live
EOF

cat > "${CHROOT_DIR}/etc/initramfs-tools/modules" <<EOF
snd_hda_intel
snd_hda_codec
snd_hda_codec_hdmi
snd_hda_codec_realtek
snd_usb_audio
snd_aloop
snd_pcm
snd_seq
snd_seq_device
snd_timer
soundcore
live-boot
squashfs
overlay
EOF

cat > "${CHROOT_DIR}/etc/modprobe.d/blacklist-kvm.conf" <<EOF
blacklist kvm
blacklist kvm_amd
blacklist kvm_intel
install kvm /bin/false
install kvm_amd /bin/false
install kvm_intel /bin/false
EOF

cat > "${CHROOT_DIR}/etc/modprobe.d/blacklist-nvidia-nouveau.conf" <<EOF
blacklist nouveau
options nouveau modeset=0
EOF

cat > "${CHROOT_DIR}/etc/modprobe.d/sound.conf" <<EOF
options snd-hda-intel power_save=0
options snd-usb-audio index=-2
EOF

chroot "${CHROOT_DIR}" update-initramfs -u -k all

mkdir -p "${CHROOT_DIR}/root" "${CHROOT_DIR}/etc/profile.d" "${CHROOT_DIR}/root/.config/lxqt" \
    "${CHROOT_DIR}/root/.config/pcmanfm-qt/lxqt" "${CHROOT_DIR}/root/.config/lxqt-panel"

for file in .bash_profile .profile .bashrc; do
    cat > "${CHROOT_DIR}/root/$file" <<'EOF'
if [[ -z "$DISPLAY" ]] && [[ "$(tty)" = "/dev/tty1" ]]; then
    exec startx
fi
EOF
done

cat > "${CHROOT_DIR}/etc/profile.d/autostartx.sh" <<'EOF'
#!/bin/sh
if [ "$(id -u)" = "0" ] && [ -z "$DISPLAY" ] && [ "$(tty)" = "/dev/tty1" ]; then
    exec startx
fi
EOF
chmod +x "${CHROOT_DIR}/etc/profile.d/autostartx.sh"

cat > "${CHROOT_DIR}/root/.xinitrc" <<'EOF'
export XDG_SESSION_TYPE=x11
export LANG=tr_TR.UTF-8
export LC_ALL=tr_TR.UTF-8
export LC_MESSAGES=tr_TR.UTF-8
export LANGUAGE=tr_TR:tr
export XDG_RUNTIME_DIR=/run/user/0
export PIPEWIRE_RUNTIME_DIR=/run/user/0
export PULSE_RUNTIME_PATH=/run/user/0/pulse
export PULSE_SERVER=unix:/run/user/0/pulse/native

mkdir -p /run/user/0
chmod 700 /run/user/0

if [ ! -S /run/user/0/pulse/native ]; then
    mkdir -p /run/user/0/pulse
    chmod 700 /run/user/0/pulse

    for proc in pipewire wireplumber pipewire-pulse; do
        pgrep -x $proc >/dev/null 2>&1 || {
            $proc >/tmp/$proc.log 2>&1 &
            for i in $(seq 1 50); do
                pgrep -x $proc >/dev/null 2>&1 && break
                [ "$proc" = "pipewire-pulse" ] && [ -S /run/user/0/pulse/native ] && break
                sleep 0.1
            done
        }
    done
fi

dbus-run-session -- sh -c '
    export XDG_SESSION_TYPE=x11
    export XDG_RUNTIME_DIR=/run/user/0
    export PIPEWIRE_RUNTIME_DIR=/run/user/0
    export PULSE_RUNTIME_PATH=/run/user/0/pulse
    export PULSE_SERVER=unix:/run/user/0/pulse/native
    export LANG=tr_TR.UTF-8
    export LC_ALL=tr_TR.UTF-8
    export LC_MESSAGES=tr_TR.UTF-8
    export LANGUAGE=tr_TR:tr

    xhost +local: >/dev/null 2>&1 || true

    gio set /root/Desktop/pusula-finans.desktop metadata::trust true 2>/dev/null || true
    gio set /root/Desktop/crystal-setup.desktop metadata::trust true 2>/dev/null || true
    gio set /root/Desktop/pusula-ai.desktop metadata::trust true 2>/dev/null || true

    xrdb -merge ~/.Xresources 2>/dev/null || true
    setxkbmap tr >/dev/null 2>&1 || true

    export PULSE_LATENCY_MSEC=30

    exec startlxqt
'
EOF

cat > "${CHROOT_DIR}/root/.config/pcmanfm-qt/lxqt/settings.conf" <<EOF
[Desktop]
DesktopShortcuts=Trash, Computer
Wallpaper=/usr/share/backgrounds/crystalos.png
WallpaperMode=stretch

[System]
Terminal=qterminal
TerminalDirCommand=qterminal -w %s
TerminalExecCommand=qterminal -e %s
EOF

if [ -f "${SCRIPT_DIR}/image/crystalos.png" ]; then
    mkdir -p "${CHROOT_DIR}/usr/share/backgrounds"
    cp "${SCRIPT_DIR}/image/crystalos.png" "${CHROOT_DIR}/usr/share/backgrounds/crystalos.png"
fi

mkdir -p "${CHROOT_DIR}/etc/xdg/pcmanfm-qt/lxqt"
cat > "${CHROOT_DIR}/etc/xdg/pcmanfm-qt/lxqt/settings.conf" <<EOF
[System]
Terminal=qterminal
TerminalDirCommand=qterminal -w %s
TerminalExecCommand=qterminal -e %s
EOF

cat > "${CHROOT_DIR}/etc/xdg/autostart/nm-tray.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=nm-tray
Exec=nm-tray
OnlyShowIn=LXQt;
Terminal=false
EOF

chroot "${CHROOT_DIR}" update-alternatives --install /usr/bin/x-terminal-emulator x-terminal-emulator /usr/bin/qterminal 50 || true
chroot "${CHROOT_DIR}" update-alternatives --set x-terminal-emulator /usr/bin/qterminal || true

cat > "${CHROOT_DIR}/root/.config/qterminal.ini" <<EOF
[General]
BackgroundTransparency=0
HideTabBarWithOneTab=false
Opacity=1
TerminalTransparency=0
Transparent=false
UseTransparency=false
transparentBackground=false
version=2.1.0

[MainWindow]
ApplicationTransparency=0

[Shortcuts]
Paste%20Clipboard=Ctrl+Shift+V
EOF

cat > "${CHROOT_DIR}/root/.config/lxqt/lxqt.conf" <<EOF
[General]
icon_theme=Papirus
EOF

cat > "${CHROOT_DIR}/root/.config/lxqt/panel.conf" <<EOF
[General]
__userfile__=true
iconTheme=Papirus

[panel1]
alignment=-1
animation-duration=0
background-color=@Variant(\0\0\0\x43\0\xff\xff\0\0\0\0\0\0\0\0)
background-image=
desktop=0
font-color=@Variant(\0\0\0\x43\0\xff\xff\0\0\0\0\0\0\0\0)
hidable=false
hide-on-overlap=false
iconSize=22
lineCount=1
lockPanel=false
opacity=100
panelSize=32
plugins=mainmenu, desktopswitch, taskbar, statusnotifier, volume, worldclock
position=Bottom
reserve-space=true
show-delay=0
visible-margin=true
width=100
width-percent=true

[mainmenu]
alignment=Left
customFont=false
type=mainmenu
ownIcon=true
icon=
showText=true
text=Menü

[desktopswitch]
alignment=Left
type=desktopswitch

[taskbar]
alignment=Left
type=taskbar

[volume]
alignment=Right
type=volume

[worldclock]
alignment=Right
type=worldclock
autoRotate=true
dateFormat=short
dateLongNames=false
dateShowDoW=false
dateShowYear=false
defaultTimeZone=Etc/GMT-0
formatType=short-timeonly
showDate=false
showTimezone=false
timeAMPM=false
timeLongNames=false
timeShowSeconds=true
timeZones=
useAdvancedManualFormat=false
EOF

chown -R root:root "${CHROOT_DIR}/root/.config/lxqt"
chmod 644 "${CHROOT_DIR}/root/.config/lxqt/panel.conf" "${CHROOT_DIR}/root/.config/lxqt/lxqt.conf" || true

rm -f "${CHROOT_DIR}/etc/xdg/autostart/lxqt-qlipper-autostart.desktop" \
    "${CHROOT_DIR}/etc/xdg/autostart/lxqt-powermanagement.desktop"

mkdir -p "${CHROOT_DIR}/etc/gtk-3.0" "${CHROOT_DIR}/etc/gtk-4.0" "${CHROOT_DIR}/etc/gtk-2.0" \
    "${CHROOT_DIR}/root/.config/openbox" "${CHROOT_DIR}/etc/xdg/lxqt" "${CHROOT_DIR}/root/.config/qt5ct" \
    "${CHROOT_DIR}/root/.config/qt6ct"

cat > "${CHROOT_DIR}/etc/gtk-3.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF

cat > "${CHROOT_DIR}/etc/gtk-4.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF

cat > "${CHROOT_DIR}/etc/gtk-2.0/gtkrc" <<EOF
gtk-theme-name="Numix"
gtk-icon-theme-name="Papirus"
EOF

cp "${SCRIPT_DIR}/crystal-setup/rc.xml" "${CHROOT_DIR}/root/.config/openbox/rc.xml"

cat > "${CHROOT_DIR}/root/.Xresources" <<EOF
Xft.dpi: 110
Xft.antialias: 1
Xft.hinting: 1
Xft.hintstyle: hintslight
Xft.rgba: rgb
EOF

cat > "${CHROOT_DIR}/etc/xdg/lxqt/session.conf" <<EOF
[General]
window_manager=openbox
EOF

cat > "${CHROOT_DIR}/etc/profile.d/qt-platformtheme.sh" <<'EOF'
#!/bin/sh
export QT_QPA_PLATFORMTHEME=qt5ct
EOF
chmod +x "${CHROOT_DIR}/etc/profile.d/qt-platformtheme.sh"

cat > "${CHROOT_DIR}/root/.config/qt5ct/qt5ct.conf" <<EOF
[Appearance]
style=gtk2
icon_theme=Papirus
font=Sans,10,-1,5,50,0,0,0,0,0
EOF

cat > "${CHROOT_DIR}/root/.config/qt6ct/qt6ct.conf" <<EOF
[Appearance]
style=gtk2
icon_theme=Papirus
font=Sans,10,-1,5,50,0,0,0,0,0
EOF

mkdir -p "${CHROOT_DIR}/opt/pusula-finans"
cp -a "${SCRIPT_DIR}/Pusula-Finans/." "${CHROOT_DIR}/opt/pusula-finans/"
chown -R root:root "${CHROOT_DIR}/opt/pusula-finans"
find "${CHROOT_DIR}/opt/pusula-finans" -type d -exec chmod 755 {} \;
find "${CHROOT_DIR}/opt/pusula-finans" -type f -exec chmod 644 {} \;
chmod +x "${CHROOT_DIR}/opt/pusula-finans/pusula_finans.py" 2>/dev/null || true

mkdir -p "${CHROOT_DIR}/usr/share/applications"
cat > "${CHROOT_DIR}/usr/share/applications/pusula-finans.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Pusula Finans
Exec=python3 /opt/pusula-finans/pusula_finans.py
Icon=/opt/pusula-finans/img/logo1.png
Terminal=false
Categories=Office;
EOF
chmod 755 "${CHROOT_DIR}/usr/share/applications/pusula-finans.desktop"
chown root:root "${CHROOT_DIR}/usr/share/applications/pusula-finans.desktop"

chroot "${CHROOT_DIR}" useradd -m -s /bin/bash vlc
chroot "${CHROOT_DIR}" passwd -d vlc

cat > "${CHROOT_DIR}/usr/share/applications/vlc.desktop" <<'EOF'
[Desktop Entry]
Version=1.0
Name=VLC media player
GenericName=Media player
Name[tr]=VLC ortam oynatıcısı
GenericName[tr]=Ortam oynatıcısı
Comment[tr]=Çoklu ortam akışlarınızı okuyun, yakalayın, yayınlayın
Exec=sudo -u vlc env DISPLAY=:0 XAUTHORITY=/root/.Xauthority LANG=tr_TR.UTF-8 LC_ALL=tr_TR.UTF-8 LANGUAGE=tr_TR:tr /usr/bin/vlc --started-from-file %U
TryExec=/usr/bin/vlc
Icon=vlc
Terminal=false
Type=Application
Categories=AudioVideo;Player;Recorder;
EOF

mkdir -p "${CHROOT_DIR}/etc/sudoers.d"
echo "root ALL=(vlc) NOPASSWD: /usr/bin/vlc" > "${CHROOT_DIR}/etc/sudoers.d/vlc"
chmod 440 "${CHROOT_DIR}/etc/sudoers.d/vlc"

mkdir -p "${CHROOT_DIR}/root/Desktop"
cp "${CHROOT_DIR}/usr/share/applications/pusula-finans.desktop" "${CHROOT_DIR}/root/Desktop/pusula-finans.desktop"
chmod +x "${CHROOT_DIR}/root/Desktop/pusula-finans.desktop"

mkdir -p "${CHROOT_DIR}/opt/crystal-setup"
cp -a "${SCRIPT_DIR}/crystal-setup/." "${CHROOT_DIR}/opt/crystal-setup"
chown -R root:root "${CHROOT_DIR}/opt/crystal-setup"

cat > "${CHROOT_DIR}/opt/crystal-setup/launch.sh" <<'EOF'
#!/bin/bash
grep -q "boot=live" /proc/cmdline && \
    exec pkexec env DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
        /opt/crystal-setup/crystal-setup.sh
EOF
chmod +x "${CHROOT_DIR}/opt/crystal-setup/launch.sh" "${CHROOT_DIR}/opt/crystal-setup/crystal-setup.sh"

cat > "${CHROOT_DIR}/root/Desktop/crystal-setup.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=CrystalOS'i Kur
Exec=/opt/crystal-setup/launch.sh
Icon=system-os-installer
Terminal=false
Categories=System;
EOF
chmod +x "${CHROOT_DIR}/root/Desktop/crystal-setup.desktop"

mkdir -p "${CHROOT_DIR}/opt/Pusula-AI"
rsync -a --exclude='v1.5/' --exclude='v2.5/' "${SCRIPT_DIR}/Pusula-AI/" "${CHROOT_DIR}/opt/Pusula-AI/"
chown -R root:root "${CHROOT_DIR}/opt/Pusula-AI"
find "${CHROOT_DIR}/opt/Pusula-AI" -type d -exec chmod 755 {} \;
find "${CHROOT_DIR}/opt/Pusula-AI" -type f -exec chmod 644 {} \;
chroot "${CHROOT_DIR}" ln -sf /opt/Pusula-AI/nvidia-llm-setup.sh /usr/local/bin/nvidia-llm-setup
chmod +x "${CHROOT_DIR}/opt/Pusula-AI/nvidia-llm-setup.sh"

cat > "${CHROOT_DIR}/usr/share/applications/pusula-ai.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Pusula AI
Path=/opt/Pusula-AI/v2.0
Exec=python3 /opt/Pusula-AI/v2.0/pusula-ai.py
Icon=/opt/Pusula-AI/img/logo.png
Terminal=true
Categories=Education;
EOF

cp "${CHROOT_DIR}/usr/share/applications/pusula-ai.desktop" "${CHROOT_DIR}/root/Desktop/pusula-ai.desktop"
chmod +x "${CHROOT_DIR}/root/Desktop/pusula-ai.desktop"

cat > "${CHROOT_DIR}/usr/lib/os-release" <<EOF
PRETTY_NAME="Crystal OS 1.3"
NAME="Debian GNU/Linux"
VERSION_ID="13"
VERSION="13 (trixie)"
VERSION_CODENAME=trixie
DEBIAN_VERSION_FULL=13.4
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
EOF

cp -a "${SCRIPT_DIR}/Lisans/." "${CHROOT_DIR}/"

cat > "${CHROOT_DIR}/HAKKINDA.txt" <<EOF
╔══════════════════════════════════════════════════════════╗
║                  CRYSTAL OS 1.3                          ║
╠══════════════════════════════════════════════════════════╣
║  Oluşturan    : Rıza Kadir ATALAY                        ║
║  Yapı Tarihi  : $(date +"%d.%m.%Y")                       ║
║  Yapı İsmi    : CRYSTAL OS 1.3 x86_64                    ║
║  Masaüstü     : LXQt + Openbox                           ║
╠══════════════════════════════════════════════════════════╣
║  Bu sistem Rıza Kadir ATALAY tarafından derlenerek       ║
║  oluşturulmuştur.                                        ║
║                                                          ║
║  CRYSTAL OS, GNU General Public License v3 (GPLv3)       ║
║  kapsamında yayımlanmaktadır.                            ║
║                                                          ║
║  Yazılım özgürce kullanılabilir, kopyalanabilir,         ║
║  değiştirilebilir ve yeniden dağıtılabilir.              ║
║                                                          ║
║  Dağıtılan değiştirilmiş sürümler GPLv3 koşullarına      ║
║  uygun olarak ilgili kaynak kodunu da sağlamalıdır.      ║
║                                                          ║
║  Lisans metni için LICENSE dosyasına bakınız.            ║
╚══════════════════════════════════════════════════════════╝
EOF

rm -f "${CHROOT_DIR}/usr/share/applications/lxqt-about.desktop" \
    "${CHROOT_DIR}/usr/share/applications/lxqt-hibernate.desktop" \
    "${CHROOT_DIR}/usr/share/applications/lxqt-lockscreen.desktop" \
    "${CHROOT_DIR}/usr/share/applications/nm-tray.desktop" \
    "${CHROOT_DIR}/usr/share/applications/qps.desktop" \
    "${CHROOT_DIR}/usr/share/applications/qt5ct.desktop" \
    "${CHROOT_DIR}/usr/share/applications/qt6ct.desktop" \
    "${CHROOT_DIR}/usr/share/applications/org.flameshot.Flameshot.desktop"

rm -rf "${CHROOT_DIR}/usr/share/doc"/* "${CHROOT_DIR}/usr/share/man"/*

DESKTOP="${CHROOT_DIR}/usr/share/applications/org.kde.falkon.desktop"
if [ -f "$DESKTOP" ]; then
    sed 's|^Exec=falkon %u|Exec=env QTWEBENGINE_DISABLE_SANDBOX=1 falkon %u|' "$DESKTOP" > "${DESKTOP}.tmp" && mv "${DESKTOP}.tmp" "$DESKTOP"
fi

umount -l "${CHROOT_DIR}/dev/pts" "${CHROOT_DIR}/dev" "${CHROOT_DIR}/proc" "${CHROOT_DIR}/sys" 2>/dev/null || true

mkdir -p "${ISO_DIR}/live" "${ISO_DIR}/.disk" "${ISO_DIR}/boot/grub/fonts" "${ISO_DIR}/EFI/BOOT"

mksquashfs "${CHROOT_DIR}" "${ISO_DIR}/live/filesystem.squashfs" -comp xz -b 1M -e boot

echo "CrystalOS" > "${ISO_DIR}/.disk/info"
echo "CRYSTALOS" > "${ISO_DIR}/.disk/cd_type"

kernelfile=$(ls -1 "${CHROOT_DIR}/boot"/vmlinuz-* 2>/dev/null | sort -V | tail -n1)
initrdfile=$(ls -1 "${CHROOT_DIR}/boot"/initrd.img-* 2>/dev/null | sort -V | tail -n1)

[ -z "$kernelfile" ] || [ -z "$initrdfile" ] && { echo "HATA: Kernel veya initrd bulunamadı!"; exit 1; }

cp "$kernelfile" "${ISO_DIR}/live/vmlinuz"
cp "$initrdfile" "${ISO_DIR}/live/initrd"

export MTOOLS_SKIP_CHECK=1
export TMPDIR="${BUILD_DIR}/tmp"
mkdir -p "${TMPDIR}"

GRUB_FONT_FILE=""
if [ -f /usr/share/grub/unicode.pf2 ]; then
    GRUB_FONT_FILE="/usr/share/grub/unicode.pf2"
elif [ -f /boot/grub/fonts/unicode.pf2 ]; then
    GRUB_FONT_FILE="/boot/grub/fonts/unicode.pf2"
else
    TTF=$(find /usr/share/fonts -name "DejaVuSansMono.ttf" 2>/dev/null | head -n1)
    [ -n "$TTF" ] && grub-mkfont -s 16 -o "${TMPDIR}/unicode.pf2" "$TTF" && GRUB_FONT_FILE="${TMPDIR}/unicode.pf2"
fi

[ -n "$GRUB_FONT_FILE" ] && cp "$GRUB_FONT_FILE" "${ISO_DIR}/boot/grub/fonts/unicode.pf2"

cat > "${TMPDIR}/grub-embed.cfg" <<'EMBEDEOF'
set timeout=5
set default=0

insmod all_video
insmod gfxterm
insmod font

search --no-floppy --set=root --label CRYSTALOS

if [ -z "$root" ]; then
    search --no-floppy --set=root --file /live/vmlinuz
fi

if loadfont ($root)/boot/grub/fonts/unicode.pf2; then
    set gfxmode=auto
    terminal_output gfxterm
fi

set menu_color_normal=white/black
set menu_color_highlight=white/magenta

menuentry "CrystalOS - Disk Üzerinden Başlatma" {
    linux /live/vmlinuz boot=live components quiet splash
    initrd /live/initrd
}

menuentry "CrystalOS - RAM Üzerinden Başlatma" {
    linux /live/vmlinuz boot=live components quiet splash toram
    initrd /live/initrd
}

menuentry "CrystalOS - Güvenli Mod" {
    linux /live/vmlinuz boot=live components
    initrd /live/initrd
}

menuentry "UEFI Donanım Yapılandırması" {
    fwsetup
}
EMBEDEOF

cp "${TMPDIR}/grub-embed.cfg" "${ISO_DIR}/boot/grub/grub.cfg"

grub-mkstandalone \
    --format=x86_64-efi \
    --output="${TMPDIR}/bootx64.efi" \
    --modules="part_gpt part_msdos iso9660 fat loopback all_video font gfxterm gfxmenu search search_fs_file search_fs_uuid search_label linux echo normal test regexp ls cat boot chain halt reboot" \
    --locales="" \
    "boot/grub/grub.cfg=${TMPDIR}/grub-embed.cfg"

EFI_SIZE_KB=$(du -k "${TMPDIR}/bootx64.efi" | cut -f1)
EFI_IMG_KB=$(( (EFI_SIZE_KB + 1024) / 1024 * 1024 + 512 ))
[ "${EFI_IMG_KB}" -lt 12288 ] && EFI_IMG_KB=12288

EFI_IMG="${TMPDIR}/efi.img"
dd if=/dev/zero of="${EFI_IMG}" bs=1k count="${EFI_IMG_KB}" status=none
mkfs.vfat -F 16 -n "CRYSTALEFI" "${EFI_IMG}"

EFI_MOUNT="${TMPDIR}/efi_mount"
mkdir -p "${EFI_MOUNT}"
mount -o loop "${EFI_IMG}" "${EFI_MOUNT}"
mkdir -p "${EFI_MOUNT}/EFI/BOOT" "${EFI_MOUNT}/boot/grub/fonts"
cp "${TMPDIR}/bootx64.efi" "${EFI_MOUNT}/EFI/BOOT/BOOTX64.EFI"
[ -n "$GRUB_FONT_FILE" ] && cp "$GRUB_FONT_FILE" "${EFI_MOUNT}/boot/grub/fonts/unicode.pf2"
umount "${EFI_MOUNT}"

cp "${TMPDIR}/bootx64.efi" "${ISO_DIR}/EFI/BOOT/BOOTX64.EFI"
cp "${EFI_IMG}" "${ISO_DIR}/boot/grub/efi.img"

xorriso -as mkisofs \
    -iso-level 3 \
    -full-iso9660-filenames \
    -volid "CRYSTALOS" \
    -publisher "Riza Kadir ATALAY" \
    -appid "Crystal OS 1.3 - Riza Kadir ATALAY tarafindan yaratildi" \
    -output "${BUILD_DIR}/crystalos.iso" \
    -eltorito-catalog EFI/BOOT/boot.cat \
    -eltorito-alt-boot \
    -e boot/grub/efi.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    "${ISO_DIR}"

echo ""
echo "================================================================"
echo "  CrystalOS başarıyla oluşturuldu!"
echo "  Konum : ${BUILD_DIR}/crystalos.iso"
echo "  Boyut : $(du -sh "${BUILD_DIR}/crystalos.iso" | cut -f1)"
echo "================================================================"
