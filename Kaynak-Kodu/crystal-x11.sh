#!/bin/bash
set -eo pipefail

if [ ! -e /dev/fd ]; then
    ln -s /proc/self/fd /dev/fd
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/crystal-build"
CHROOT_DIR="${BUILD_DIR}/chroot"
ISO_DIR="${BUILD_DIR}/iso"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${CHROOT_DIR}"

debootstrap --arch=amd64 --include=locales trixie "${CHROOT_DIR}" http://deb.debian.org/debian

mount --bind /dev     "${CHROOT_DIR}/dev"
mount --bind /dev/pts "${CHROOT_DIR}/dev/pts"
mount --bind /proc    "${CHROOT_DIR}/proc"
mount --bind /sys     "${CHROOT_DIR}/sys"

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

echo "en_US.UTF-8 UTF-8" >> "${CHROOT_DIR}/etc/locale.gen"
echo "tr_TR.UTF-8 UTF-8" >> "${CHROOT_DIR}/etc/locale.gen"
chroot "${CHROOT_DIR}" locale-gen

cat > "${CHROOT_DIR}/etc/default/locale" <<EOF
LANG=tr_TR.UTF-8
LC_ALL=tr_TR.UTF-8
LC_MESSAGES=tr_TR.UTF-8
LANGUAGE=tr_TR:tr
EOF

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    --no-install-recommends \
    linux-image-amd64 \
    linux-headers-amd64 \
    build-essential \
    dkms \
    initramfs-tools \
    squashfs-tools \
    live-boot \
    live-boot-initramfs-tools \
    live-config \
    live-config-systemd \
    xorriso \
    systemd-sysv \
    network-manager \
    nm-tray \
    wireless-tools \
    wpasupplicant \
    dbus \
    dbus-x11 \
    sudo \
    keyboard-configuration \
    console-setup \
    htop \
    fastfetch \
    vlc \
    vlc-l10n \
    debootstrap \
    cmatrix \
    gdebi \
    qbittorrent \
    flameshot \
    qemu-utils \
    libguestfs-tools \
    guestfs-tools \
    fuse3 \
    gvfs-fuse \
    kpartx \
    fdisk \
    util-linux \
    soundconverter \
    inkscape

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    --no-install-recommends \
    task-lxqt-desktop \
    lxqt-core \
    lxqt-config \
    lxqt-admin \
    lxqt-qtplugin \
    lxqt-powermanagement \
    lxqt-notificationd \
    lxqt-policykit \
    pcmanfm-qt \
    pcmanfm-qt-l10n \
    libfm-qt-l10n \
    qterminal \
    openbox \
    obconf-qt \
    git \
    curl

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    --no-install-recommends \
    xserver-xorg \
    xserver-xorg-core \
    xserver-xorg-input-all \
    xserver-xorg-video-all \
    xinit \
    x11-xserver-utils

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    --no-install-recommends \
    gparted \
    parted \
    eject \
    e2fsprogs \
    dosfstools \
    ntfs-3g \
    exfatprogs \
    xfsprogs \
    rsync \
    grub-efi-amd64-bin \
    grub-common \
    mtools \
    efibootmgr \
    yad \
    papirus-icon-theme \
    fonts-dejavu \
    fonts-liberation \
    fonts-font-awesome \
    pulseaudio \
    pavucontrol-qt \
    gvfs \
    gvfs-backends \
    udisks2 \
    upower \
    polkitd \
    pkexec \
    numix-gtk-theme \
    qt5ct \
    qt6ct

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
    if git ls-remote https://github.com/numixproject/numix-gtk-theme.git >/dev/null 2>&1; then
        git clone --depth=1 https://github.com/numixproject/numix-gtk-theme.git || true
        if [ -d numix-gtk-theme/openbox-3 ]; then
            mkdir -p /usr/share/themes/Numix
            cp -a numix-gtk-theme/openbox-3 /usr/share/themes/Numix/openbox-3 || true
        elif [ -d numix-gtk-theme/Openbox-3 ]; then
            mkdir -p /usr/share/themes/Numix
            cp -a numix-gtk-theme/Openbox-3 /usr/share/themes/Numix/openbox-3 || true
        else
            for d in numix-gtk-theme/*openbox*; do
                if [ -d "$d" ]; then
                    mkdir -p /usr/share/themes/Numix
                    cp -a "$d" /usr/share/themes/Numix/ || true
                fi
            done
        fi
    fi
fi
'

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    firmware-linux \
    firmware-linux-free \
    firmware-linux-nonfree \
    firmware-misc-nonfree

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    lxqt-about-l10n \
    lxqt-config-l10n \
    lxqt-session-l10n \
    lxqt-panel-l10n \
    lxqt-policykit-l10n \
    liblxqt-l10n \
    librsvg2-common \
    libqt5svg5

DEBIAN_FRONTEND=noninteractive TMPDIR=/tmp chroot "${CHROOT_DIR}" apt-get install -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    python3 \
    python3-pip \
    python3-venv \
    falkon || true

chroot "${CHROOT_DIR}" pip3 install --break-system-packages --root-user-action=ignore \
    yfinance \
    mplfinance \
    scikit-learn \
    PyQt6 \
    sentencepiece \
    requests

chroot "${CHROOT_DIR}" pip3 install --break-system-packages --root-user-action=ignore \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

chroot "${CHROOT_DIR}" systemctl disable sddm     2>/dev/null || true
chroot "${CHROOT_DIR}" systemctl disable lightdm  2>/dev/null || true
chroot "${CHROOT_DIR}" systemctl disable gdm      2>/dev/null || true
chroot "${CHROOT_DIR}" systemctl mask    sddm     2>/dev/null || true
chroot "${CHROOT_DIR}" systemctl mask    lightdm  2>/dev/null || true
chroot "${CHROOT_DIR}" systemctl mask    gdm      2>/dev/null || true

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
EOF

rm -f "${CHROOT_DIR}/lib/live/config/1160-openssh-server" 2>/dev/null || true
rm -f "${CHROOT_DIR}/lib/live/config/1170-user-setup"     2>/dev/null || true

chroot "${CHROOT_DIR}" systemctl enable getty@tty1

mkdir -p "${CHROOT_DIR}/root"
mkdir -p "${CHROOT_DIR}/root/.config/lxqt"
mkdir -p "${CHROOT_DIR}/root/.config/pcmanfm-qt/lxqt"
mkdir -p "${CHROOT_DIR}/root/.config/lxqt-panel"

cat > "${CHROOT_DIR}/root/.bash_profile" <<'EOF'
if [[ -z $DISPLAY ]] && [[ $(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF

cat > "${CHROOT_DIR}/root/.profile" <<'EOF'
if [[ -z $DISPLAY ]] && [[ $(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF

cat > "${CHROOT_DIR}/root/.bashrc" <<'EOF'
if [[ -z $DISPLAY ]] && [[ $(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF

mkdir -p "${CHROOT_DIR}/etc/profile.d"
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
xhost +local: &
gio set /root/Desktop/pusula-finans.desktop  metadata::trust true 2>/dev/null || true
gio set /root/Desktop/crystal-setup.desktop  metadata::trust true 2>/dev/null || true
gio set /root/Desktop/pusula-ai.desktop  metadata::trust true 2>/dev/null || true
pulseaudio --start &
xrdb -merge ~/.Xresources
setxkbmap tr &
exec startlxqt
EOF

mkdir -p "${CHROOT_DIR}/root/.config/pcmanfm-qt/lxqt"
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
cat > "${CHROOT_DIR}/etc/xdg/pcmanfm-qt/lxqt/settings.conf" <<'EOF'
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

chroot "${CHROOT_DIR}" update-alternatives --install \
    /usr/bin/x-terminal-emulator x-terminal-emulator /usr/bin/qterminal 50 || true
chroot "${CHROOT_DIR}" update-alternatives --set \
    x-terminal-emulator /usr/bin/qterminal || true

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

mkdir -p "${CHROOT_DIR}/root/.config/lxqt"
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
defaultTimeZone=Europe/Istanbul
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
chmod 644 "${CHROOT_DIR}/root/.config/lxqt/panel.conf" || true
chmod 644 "${CHROOT_DIR}/root/.config/lxqt/lxqt.conf"  || true

rm -f "${CHROOT_DIR}/etc/xdg/autostart/lxqt-qlipper-autostart.desktop"
rm -f "${CHROOT_DIR}/etc/xdg/autostart/lxqt-powermanagement.desktop"

chroot "${CHROOT_DIR}" systemctl enable NetworkManager

cat > "${CHROOT_DIR}/etc/initramfs-tools/initramfs.conf" <<EOF
MODULES=most
BUSYBOX=y
COMPRESS=gzip
BOOT=live
EOF

mkdir -p "${CHROOT_DIR}/etc/initramfs-tools/conf.d"
cat > "${CHROOT_DIR}/etc/initramfs-tools/conf.d/live.conf" <<EOF
export LIVE=true
export BOOT=live
EOF

echo "live-boot" >> "${CHROOT_DIR}/etc/initramfs-tools/modules"
echo "squashfs"  >> "${CHROOT_DIR}/etc/initramfs-tools/modules"
echo "overlay"   >> "${CHROOT_DIR}/etc/initramfs-tools/modules"

mkdir -p "${CHROOT_DIR}/etc/modprobe.d"
cat > "${CHROOT_DIR}/etc/modprobe.d/blacklist-kvm.conf" <<'EOF'
blacklist kvm
blacklist kvm_amd
blacklist kvm_intel
install kvm /bin/false
install kvm_amd /bin/false
install kvm_intel /bin/false
EOF

chroot "${CHROOT_DIR}" update-initramfs -u -k all

mkdir -p "${CHROOT_DIR}/etc/gtk-3.0"
cat > "${CHROOT_DIR}/etc/gtk-3.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF

mkdir -p "${CHROOT_DIR}/etc/gtk-4.0"
cat > "${CHROOT_DIR}/etc/gtk-4.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF

mkdir -p "${CHROOT_DIR}/etc/gtk-2.0"
cat > "${CHROOT_DIR}/etc/gtk-2.0/gtkrc" <<EOF
gtk-theme-name="Numix"
gtk-icon-theme-name="Papirus"
EOF

mkdir -p "${CHROOT_DIR}/root/.config/openbox"
cat > "${CHROOT_DIR}/root/.config/openbox/rc.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc" xmlns:xi="http://www.w3.org/2001/XInclude">
<resistance>
  <strength>10</strength>
  <screen_edge_strength>20</screen_edge_strength>
</resistance>
<focus>
  <focusNew>yes</focusNew>
  <followMouse>no</followMouse>
  <focusLast>yes</focusLast>
  <underMouse>no</underMouse>
  <focusDelay>200</focusDelay>
  <raiseOnFocus>no</raiseOnFocus>
</focus>
<placement>
  <policy>Smart</policy>
  <center>yes</center>
  <monitor>Primary</monitor>
  <primaryMonitor>1</primaryMonitor>
</placement>
<theme>
  <name>Numix</name>
  <titleLayout>NLIMC</titleLayout>
  <keepBorder>yes</keepBorder>
  <animateIconify>yes</animateIconify>
  <font place="ActiveWindow">
    <name>DejaVu Sans</name><size>10</size><weight>Bold</weight><slant>Normal</slant>
  </font>
  <font place="InactiveWindow">
    <name>DejaVu Sans</name><size>10</size><weight>Bold</weight><slant>Normal</slant>
  </font>
  <font place="MenuHeader">
    <name>DejaVu Sans</name><size>10</size><weight>Bold</weight><slant>Normal</slant>
  </font>
  <font place="MenuItem">
    <name>DejaVu Sans</name><size>10</size><weight>Normal</weight><slant>Normal</slant>
  </font>
  <font place="ActiveOnScreenDisplay">
    <name>DejaVu Sans</name><size>10</size><weight>Bold</weight><slant>Normal</slant>
  </font>
  <font place="InactiveOnScreenDisplay">
    <name>DejaVu Sans</name><size>10</size><weight>Bold</weight><slant>Normal</slant>
  </font>
</theme>
<desktops>
  <number>4</number>
  <firstdesk>1</firstdesk>
  <names/>
  <popupTime>875</popupTime>
</desktops>
<resize>
  <drawContents>yes</drawContents>
  <popupShow>Nonpixel</popupShow>
  <popupPosition>Center</popupPosition>
  <popupFixedPosition><x>10</x><y>10</y></popupFixedPosition>
</resize>
<margins>
  <top>0</top><bottom>0</bottom><left>0</left><right>0</right>
</margins>
<dock>
  <position>TopLeft</position>
  <floatingX>0</floatingX><floatingY>0</floatingY>
  <noStrut>no</noStrut>
  <stacking>Above</stacking>
  <direction>Vertical</direction>
  <autoHide>no</autoHide>
  <hideDelay>300</hideDelay><showDelay>300</showDelay>
  <moveButton>Middle</moveButton>
</dock>
<keyboard>
  <chainQuitKey>C-g</chainQuitKey>
  <keybind key="C-A-Left">
    <action name="GoToDesktop"><to>left</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="C-A-Right">
    <action name="GoToDesktop"><to>right</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="C-A-Up">
    <action name="GoToDesktop"><to>up</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="C-A-Down">
    <action name="GoToDesktop"><to>down</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="S-A-Left">
    <action name="SendToDesktop"><to>left</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="S-A-Right">
    <action name="SendToDesktop"><to>right</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="S-A-Up">
    <action name="SendToDesktop"><to>up</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="S-A-Down">
    <action name="SendToDesktop"><to>down</to><wrap>no</wrap></action>
  </keybind>
  <keybind key="W-F1"><action name="GoToDesktop"><to>1</to></action></keybind>
  <keybind key="W-F2"><action name="GoToDesktop"><to>2</to></action></keybind>
  <keybind key="W-F3"><action name="GoToDesktop"><to>3</to></action></keybind>
  <keybind key="W-F4"><action name="GoToDesktop"><to>4</to></action></keybind>
  <keybind key="W-d"><action name="ToggleShowDesktop"/></keybind>
  <keybind key="A-F4"><action name="Close"/></keybind>
  <keybind key="A-Escape">
    <action name="Lower"/>
    <action name="FocusToBottom"/>
    <action name="Unfocus"/>
  </keybind>
  <keybind key="A-space">
    <action name="ShowMenu"><menu>client-menu</menu></action>
  </keybind>
  <keybind key="A-Print">
    <action name="Execute"><command>scrot -s</command></action>
  </keybind>
  <keybind key="A-Tab">
    <action name="NextWindow">
      <finalactions>
        <action name="Focus"/><action name="Raise"/><action name="Unshade"/>
      </finalactions>
    </action>
  </keybind>
  <keybind key="A-S-Tab">
    <action name="PreviousWindow">
      <finalactions>
        <action name="Focus"/><action name="Raise"/><action name="Unshade"/>
      </finalactions>
    </action>
  </keybind>
  <keybind key="C-A-Tab">
    <action name="NextWindow">
      <panels>yes</panels><desktop>yes</desktop>
      <finalactions>
        <action name="Focus"/><action name="Raise"/><action name="Unshade"/>
      </finalactions>
    </action>
  </keybind>
  <keybind key="W-S-Right">
    <action name="DirectionalCycleWindows"><direction>right</direction></action>
  </keybind>
  <keybind key="W-S-Left">
    <action name="DirectionalCycleWindows"><direction>left</direction></action>
  </keybind>
  <keybind key="W-S-Up">
    <action name="DirectionalCycleWindows"><direction>up</direction></action>
  </keybind>
  <keybind key="W-S-Down">
    <action name="DirectionalCycleWindows"><direction>down</direction></action>
  </keybind>
  <keybind key="Print">
    <action name="Execute"><command>flameshot gui</command></action>
  </keybind>
  <keybind key="C-A-t">
    <action name="Execute"><command>qterminal</command></action>
  </keybind>
</keyboard>
<mouse>
  <dragThreshold>1</dragThreshold>
  <doubleClickTime>500</doubleClickTime>
  <screenEdgeWarpTime>400</screenEdgeWarpTime>
  <screenEdgeWarpMouse>false</screenEdgeWarpMouse>
  <context name="Frame">
    <mousebind button="A-Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="A-Left" action="Drag"><action name="Move"/></mousebind>
    <mousebind button="A-Right" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="A-Right" action="Drag"><action name="Resize"/></mousebind>
    <mousebind button="A-Middle" action="Press">
      <action name="Lower"/><action name="FocusToBottom"/><action name="Unfocus"/>
    </mousebind>
    <mousebind button="A-Up" action="Click">
      <action name="GoToDesktop"><to>previous</to></action>
    </mousebind>
    <mousebind button="A-Down" action="Click">
      <action name="GoToDesktop"><to>next</to></action>
    </mousebind>
  </context>
  <context name="Titlebar">
    <mousebind button="Left" action="Drag"><action name="Move"/></mousebind>
    <mousebind button="Left" action="DoubleClick"><action name="ToggleMaximize"/></mousebind>
  </context>
  <context name="Titlebar Top Right Bottom Left TLCorner TRCorner BRCorner BLCorner">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="Middle" action="Press">
      <action name="Lower"/><action name="FocusToBottom"/><action name="Unfocus"/>
    </mousebind>
    <mousebind button="Right" action="Press">
      <action name="Focus"/><action name="Raise"/>
      <action name="ShowMenu"><menu>client-menu</menu></action>
    </mousebind>
  </context>
  <context name="Top">
    <mousebind button="Left" action="Drag">
      <action name="Resize"><edge>top</edge></action>
    </mousebind>
  </context>
  <context name="Left">
    <mousebind button="Left" action="Drag">
      <action name="Resize"><edge>left</edge></action>
    </mousebind>
  </context>
  <context name="Right">
    <mousebind button="Left" action="Drag">
      <action name="Resize"><edge>right</edge></action>
    </mousebind>
  </context>
  <context name="Bottom">
    <mousebind button="Left" action="Drag">
      <action name="Resize"><edge>bottom</edge></action>
    </mousebind>
    <mousebind button="Right" action="Press">
      <action name="Focus"/><action name="Raise"/>
      <action name="ShowMenu"><menu>client-menu</menu></action>
    </mousebind>
  </context>
  <context name="TRCorner BRCorner TLCorner BLCorner">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="Left" action="Drag"><action name="Resize"/></mousebind>
  </context>
  <context name="Client">
    <mousebind button="Left"   action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Middle" action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Right"  action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
  </context>
  <context name="Icon">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
      <action name="ShowMenu"><menu>client-menu</menu></action>
    </mousebind>
    <mousebind button="Right" action="Press">
      <action name="Focus"/><action name="Raise"/>
      <action name="ShowMenu"><menu>client-menu</menu></action>
    </mousebind>
  </context>
  <context name="AllDesktops">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="Left" action="Click"><action name="ToggleOmnipresent"/></mousebind>
  </context>
  <context name="Iconify">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="Left" action="Click"><action name="Iconify"/></mousebind>
  </context>
  <context name="Maximize">
    <mousebind button="Left"   action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Middle" action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Right"  action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Left"   action="Click"><action name="ToggleMaximize"/></mousebind>
    <mousebind button="Middle" action="Click">
      <action name="ToggleMaximize"><direction>vertical</direction></action>
    </mousebind>
    <mousebind button="Right" action="Click">
      <action name="ToggleMaximize"><direction>horizontal</direction></action>
    </mousebind>
  </context>
  <context name="Close">
    <mousebind button="Left" action="Press">
      <action name="Focus"/><action name="Raise"/>
    </mousebind>
    <mousebind button="Left" action="Click"><action name="Close"/></mousebind>
  </context>
  <context name="Desktop">
    <mousebind button="Up"   action="Click"><action name="GoToDesktop"><to>previous</to></action></mousebind>
    <mousebind button="Down" action="Click"><action name="GoToDesktop"><to>next</to></action></mousebind>
    <mousebind button="Left"  action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
    <mousebind button="Right" action="Press"><action name="Focus"/><action name="Raise"/></mousebind>
  </context>
  <context name="Root">
    <mousebind button="Middle" action="Press">
      <action name="ShowMenu"><menu>client-list-combined-menu</menu></action>
    </mousebind>
    <mousebind button="Right" action="Press">
      <action name="ShowMenu"><menu>root-menu</menu></action>
    </mousebind>
  </context>
  <context name="MoveResize">
    <mousebind button="Up"   action="Click"><action name="GoToDesktop"><to>previous</to></action></mousebind>
    <mousebind button="Down" action="Click"><action name="GoToDesktop"><to>next</to></action></mousebind>
  </context>
</mouse>
<menu>
  <file>/var/lib/openbox/debian-menu.xml</file>
  <file>menu.xml</file>
  <hideDelay>200</hideDelay>
  <middle>no</middle>
  <submenuShowDelay>100</submenuShowDelay>
  <submenuHideDelay>400</submenuHideDelay>
  <showIcons>yes</showIcons>
  <manageDesktops>yes</manageDesktops>
</menu>
<applications/>
</openbox_config>
EOF

cat > "${CHROOT_DIR}/root/.Xresources" <<EOF
Xft.dpi: 110
Xft.antialias: 1
Xft.hinting: 1
Xft.hintstyle: hintslight
Xft.rgba: rgb
EOF

mkdir -p "${CHROOT_DIR}/etc/xdg/lxqt"
cat > "${CHROOT_DIR}/etc/xdg/lxqt/session.conf" <<EOF
[General]
window_manager=openbox
EOF

cat > "${CHROOT_DIR}/etc/profile.d/qt-platformtheme.sh" <<'EOF'
#!/bin/sh
export QT_QPA_PLATFORMTHEME=qt5ct
EOF
chmod +x "${CHROOT_DIR}/etc/profile.d/qt-platformtheme.sh"

mkdir -p "${CHROOT_DIR}/root/.config/qt5ct"
cat > "${CHROOT_DIR}/root/.config/qt5ct/qt5ct.conf" <<EOF
[Appearance]
style=gtk2
icon_theme=Papirus
font=Sans,10,-1,5,50,0,0,0,0,0
EOF

mkdir -p "${CHROOT_DIR}/root/.config/qt6ct"
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
echo "root ALL=(vlc) NOPASSWD: /usr/bin/vlc" \
    > "${CHROOT_DIR}/etc/sudoers.d/vlc"
chmod 440 "${CHROOT_DIR}/etc/sudoers.d/vlc"

mkdir -p "${CHROOT_DIR}/root/Desktop"
cp "${CHROOT_DIR}/usr/share/applications/pusula-finans.desktop" \
    "${CHROOT_DIR}/root/Desktop/pusula-finans.desktop"
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
chmod +x "${CHROOT_DIR}/opt/crystal-setup/launch.sh"

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
rsync -a --exclude='venv/' --exclude='python_venv_olusturma.txt' "${SCRIPT_DIR}/Pusula-AI/" "${CHROOT_DIR}/opt/Pusula-AI/"
chown -R root:root "${CHROOT_DIR}/opt/Pusula-AI"
find "${CHROOT_DIR}/opt/Pusula-AI" -type d -exec chmod 755 {} \;
find "${CHROOT_DIR}/opt/Pusula-AI" -type f -exec chmod 644 {} \;
cat > "${CHROOT_DIR}/usr/share/applications/pusula-ai.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Pusula AI
Path=/opt/Pusula-AI/v1.5
Exec=python3 /opt/Pusula-AI/v1.5/pusula-ai.py
Icon=/opt/Pusula-AI/img/logo.png
Terminal=true
Categories=Education;
EOF

cp "${CHROOT_DIR}/usr/share/applications/pusula-ai.desktop" \
    "${CHROOT_DIR}/root/Desktop/pusula-ai.desktop"
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
║  Yapı Tarihi  : $(date +"%d.%m.%Y")                               ║
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

rm -f "${CHROOT_DIR}/usr/share/applications/lxqt-about.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/lxqt-hibernate.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/lxqt-lockscreen.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/nm-tray.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/qps.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/qt5ct.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/qt6ct.desktop"
rm -f "${CHROOT_DIR}/usr/share/applications/org.flameshot.Flameshot.desktop"

DESKTOP="${CHROOT_DIR}/usr/share/applications/org.kde.falkon.desktop"
if [ -f "$DESKTOP" ]; then
    sed 's|^Exec=falkon %u|Exec=env QTWEBENGINE_DISABLE_SANDBOX=1 falkon %u|' \
        "$DESKTOP" > "${DESKTOP}.tmp" && mv "${DESKTOP}.tmp" "$DESKTOP"
fi

umount -l "${CHROOT_DIR}/dev/pts" 2>/dev/null || true
umount -l "${CHROOT_DIR}/dev"     2>/dev/null || true
umount -l "${CHROOT_DIR}/proc"    2>/dev/null || true
umount -l "${CHROOT_DIR}/sys"     2>/dev/null || true

mkdir -p "${ISO_DIR}/live"

mksquashfs "${CHROOT_DIR}" "${ISO_DIR}/live/filesystem.squashfs" \
    -comp xz -b 1M -e boot

mkdir -p "${ISO_DIR}/.disk"
echo "CrystalOS"  > "${ISO_DIR}/.disk/info"
echo "CRYSTALOS"  > "${ISO_DIR}/.disk/cd_type"

kernelfile=$(ls -1 "${CHROOT_DIR}/boot"/vmlinuz-* 2>/dev/null | sort -V | tail -n1)
initrdfile=$(ls -1 "${CHROOT_DIR}/boot"/initrd.img-* 2>/dev/null | sort -V | tail -n1)

if [ -z "$kernelfile" ] || [ -z "$initrdfile" ]; then
    echo "HATA: Kernel veya initrd bulunamadı!"
    exit 1
fi

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
    if [ -n "$TTF" ]; then
        grub-mkfont -s 16 -o "${TMPDIR}/unicode.pf2" "$TTF"
        GRUB_FONT_FILE="${TMPDIR}/unicode.pf2"
    fi
fi

mkdir -p "${ISO_DIR}/boot/grub/fonts"
[ -n "$GRUB_FONT_FILE" ] && cp "$GRUB_FONT_FILE" "${ISO_DIR}/boot/grub/fonts/unicode.pf2"
cat > "${TMPDIR}/grub-embed.cfg" <<'EMBEDEOF'
# -------------------------------------------------
# CrystalOS GRUB Yapılandırması
# -------------------------------------------------

set timeout=5
set default=0

# -------------------------------------------------
# Gerekli Modüller
# -------------------------------------------------

insmod all_video
insmod gfxterm
insmod font

# -------------------------------------------------
# ISO / Live Sistem Konumunu Bul
# -------------------------------------------------

search --no-floppy --set=root --label CRYSTALOS

if [ -z "$root" ]; then
    search --no-floppy --set=root --file /live/vmlinuz
fi

# -------------------------------------------------
# Grafik Terminal ve Font
# -------------------------------------------------

if loadfont ($root)/boot/grub/fonts/unicode.pf2; then
    set gfxmode=auto
    terminal_output gfxterm
fi

# -------------------------------------------------
# Menü Renkleri
# -------------------------------------------------

set menu_color_normal=white/black
set menu_color_highlight=white/magenta

# -------------------------------------------------
# Menü Girdileri
# -------------------------------------------------

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
mkdir -p "${ISO_DIR}/boot/grub"
cp "${TMPDIR}/grub-embed.cfg" "${ISO_DIR}/boot/grub/grub.cfg"

grub-mkstandalone \
    --format=x86_64-efi \
    --output="${TMPDIR}/bootx64.efi" \
    --modules="part_gpt part_msdos iso9660 fat loopback all_video font gfxterm gfxmenu search search_fs_file search_fs_uuid search_label linux echo normal test regexp ls cat boot chain halt reboot" \
    --locales="" \
    "boot/grub/grub.cfg=${TMPDIR}/grub-embed.cfg"

EFI_SIZE_KB=$(du -k "${TMPDIR}/bootx64.efi" | cut -f1)
EFI_IMG_KB=$(( (EFI_SIZE_KB + 1024) / 1024 * 1024 + 512 ))
[ "${EFI_IMG_KB}" -lt 16384 ] && EFI_IMG_KB=16384

EFI_IMG="${TMPDIR}/efi.img"
dd if=/dev/zero of="${EFI_IMG}" bs=1k count="${EFI_IMG_KB}" status=none
mkfs.vfat -F 16 -n "CRYSTALEFI" "${EFI_IMG}"

EFI_MOUNT="${TMPDIR}/efi_mount"
mkdir -p "${EFI_MOUNT}"
mount -o loop "${EFI_IMG}" "${EFI_MOUNT}"

mkdir -p "${EFI_MOUNT}/EFI/BOOT"
mkdir -p "${EFI_MOUNT}/boot/grub/fonts"

cp "${TMPDIR}/bootx64.efi" "${EFI_MOUNT}/EFI/BOOT/BOOTX64.EFI"
[ -n "$GRUB_FONT_FILE" ] && cp "$GRUB_FONT_FILE" "${EFI_MOUNT}/boot/grub/fonts/unicode.pf2" || true

umount "${EFI_MOUNT}"

mkdir -p "${ISO_DIR}/EFI/BOOT"
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
