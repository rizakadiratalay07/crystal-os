#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOUNT="/mnt/crystalos"
IKON="/usr/share/icons/Papirus/64x64/apps/system-os-installer.svg"
YAD_BASE=(yad --window-icon="$IKON" --text-align=center)
LOGFILE="/tmp/crystalos-install.log"

die() {
    "${YAD_BASE[@]}" --error --title="Hata" --text="$1" >/dev/null 2>&1 || true
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Gerekli komut bulunamadı: $1"
}

unmount_disk_tree() {
    local disk="$1"
    local parts
    parts=$(lsblk -lnpo NAME "$disk" 2>/dev/null | tail -n +2 || true)

    if [ -n "${parts:-}" ]; then
        while IFS= read -r dev; do
            [ -n "$dev" ] || continue
            umount -l "$dev" 2>/dev/null || true
        done <<< "$parts"
    fi
}

cleanup_mounts() {
    sync || true
    if mountpoint -q "$MOUNT/sys/firmware/efi/efivars" 2>/dev/null; then
        umount -l "$MOUNT/sys/firmware/efi/efivars" 2>/dev/null || true
    fi
    for d in dev/pts dev proc sys run boot/efi; do
        if mountpoint -q "$MOUNT/$d" 2>/dev/null; then
            umount -l "$MOUNT/$d" 2>/dev/null || true
        fi
    done
    if mountpoint -q "$MOUNT" 2>/dev/null; then
        umount -l "$MOUNT" 2>/dev/null || true
    fi
}

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    die "Root yetkisi gerekli!\n\nsudo ./crystalos-install.sh"
fi

for c in yad lsblk awk wipefs parted partprobe udevadm mkfs.ext4 mkfs.fat rsync chroot grub-install update-initramfs lsinitramfs blkid useradd chpasswd mount umount depmod; do
    need_cmd "$c"
done

"${YAD_BASE[@]}" --info --title="CrystalOS Kurulum" --width=420 \
    --text="<b>CrystalOS Kurulum Sihirbazına Hoş Geldiniz</b>\n\nBu sihirbaz CrystalOS'u bilgisayarınıza kuracaktır.\n\nUYARI: Seçilen diskteki tüm veriler silinecektir!\nDevam etmeden önce verilerinizi yedekleyin." \
    --button="Devam Et:0" --button="İptal:1" || exit 0

while true; do
    DISK=$(
        lsblk -dnpo NAME,SIZE,TYPE |
        awk '$3=="disk"{print $1; print $2}' |
        "${YAD_BASE[@]}" --list --title="Disk Seçimi" --width=500 --height=300 \
            --text="UYARI: Seçilen diskteki TÜM VERİLER SİLİNECEKTİR!\nKurulum diski seçin:" \
            --column="Disk" --column="Boyut" \
            --button="Devam Et:0" --button="İptal:1" \
            --print-column=1
    ) || exit 0

    DISK="${DISK%%|*}"
    DISK="$(echo "$DISK" | xargs)"

    if [[ -n "$DISK" && -b "$DISK" ]]; then
        break
    fi

    "${YAD_BASE[@]}" --warning --title="Uyarı" --text="Lütfen geçerli bir disk seçin!" --button="Tamam:0" >/dev/null
done

SONUC=$(
    "${YAD_BASE[@]}" --form --title="Kullanıcı Bilgileri" --width=380 \
        --text="Kurulum için kullanıcı bilgilerini girin:" \
        --field="Kullanıcı Adı:" "" \
        --field="Bilgisayar Adı:" "crystal-pc" \
        --field="Şifre:H" "" \
        --field="Şifre Tekrar:H" "" \
        --button="Devam Et:0" --button="İptal:1"
) || exit 0

USERNAME=$(echo "$SONUC" | cut -d'|' -f1)
HOSTNAME=$(echo "$SONUC"  | cut -d'|' -f2)
SIFRE=$(echo "$SONUC"     | cut -d'|' -f3)
SIFRE2=$(echo "$SONUC"    | cut -d'|' -f4)

[[ -z "$USERNAME" || ! "$USERNAME" =~ ^[a-z][a-z0-9_]*$ ]] && die "Geçersiz kullanıcı adı!"
[ "${#SIFRE}" -lt 6 ] && die "Şifre en az 6 karakter olmalıdır!"
[ "$SIFRE" != "$SIFRE2" ] && die "Şifreler eşleşmiyor!"

"${YAD_BASE[@]}" --question --title="Kurulumu Başlat" --width=400 \
    --text="<b>Kurulum Özeti</b>\n\nHedef Disk:       $DISK\nKullanıcı:        $USERNAME\nBilgisayar Adı:   $HOSTNAME\n\nUYARI: <b>$DISK</b> diskindeki tüm veriler silinecektir!" \
    --button="Kurulumu Başlat:0" --button="İptal:1" || exit 0

(
set -Eeuo pipefail
exec > >(tee -a "$LOGFILE") 2>&1
trap cleanup_mounts EXIT

echo "5"
mkdir -p "$MOUNT"

unmount_disk_tree "$DISK" || true
wipefs -a "$DISK"

echo "10"
if [ -d /sys/firmware/efi ]; then
    UEFI=1
    parted -s "$DISK" mklabel gpt
    parted -s "$DISK" mkpart ESP fat32 1MiB 256MiB
    parted -s "$DISK" set 1 esp on
    parted -s "$DISK" mkpart primary ext4 256MiB 100%
else
    UEFI=0
    parted -s "$DISK" mklabel msdos
    parted -s "$DISK" mkpart primary ext4 1MiB 100%
fi

partprobe "$DISK" || true
udevadm settle

if [[ "${DISK: -1}" =~ [0-9] ]]; then
    if [ "$UEFI" -eq 1 ]; then
        P1="${DISK}p1"
        P2="${DISK}p2"
    else
        P2="${DISK}p1"
    fi
else
    if [ "$UEFI" -eq 1 ]; then
        P1="${DISK}1"
        P2="${DISK}2"
    else
        P2="${DISK}1"
    fi
fi

for _ in {1..20}; do
    [ "$UEFI" -eq 1 ] && [ -b "$P1" ] && [ -b "$P2" ] && break
    [ "$UEFI" -eq 0 ] && [ -b "$P2" ] && break
    sleep 1
    partprobe "$DISK" || true
    udevadm settle
done

if [ "$UEFI" -eq 1 ]; then
    [ -b "$P1" ] || die "ESP bölümü bulunamadı: $P1"
    [ -b "$P2" ] || die "Root bölümü bulunamadı: $P2"
else
    [ -b "$P2" ] || die "Root bölümü bulunamadı: $P2"
fi

echo "20"
if [ "$UEFI" -eq 1 ]; then
    mkfs.fat -F32 "$P1"
    mkfs.ext4 -F "$P2"
else
    mkfs.ext4 -F "$P2"
fi

echo "28"
mount "$P2" "$MOUNT"
if [ "$UEFI" -eq 1 ]; then
    mkdir -p "$MOUNT/boot/efi"
    mount "$P1" "$MOUNT/boot/efi"
fi

echo "32"
rsync -aAXH --numeric-ids \
    --exclude=/dev \
    --exclude=/proc \
    --exclude=/sys \
    --exclude=/run \
    --exclude=/mnt \
    --exclude=/tmp \
    --exclude=/media \
    --exclude=/cdrom \
    --exclude=/lost+found \
    / "$MOUNT"/

mkdir -p "$MOUNT/tmp" "$MOUNT/run" "$MOUNT/var/tmp"
chmod 1777 "$MOUNT/tmp" "$MOUNT/var/tmp"
chmod 755 "$MOUNT/run"

echo "Live kalıntıları temizleniyor..."

chroot "$MOUNT" dpkg --purge --force-all \
    live-boot live-boot-initramfs-tools live-tools casper \
    overlayroot 2>/dev/null || true

for live_path in \
    "$MOUNT/usr/share/initramfs-tools/scripts/live" \
    "$MOUNT/usr/share/initramfs-tools/scripts/init-top/live" \
    "$MOUNT/usr/share/initramfs-tools/scripts/init-premount/live" \
    "$MOUNT/usr/share/initramfs-tools/scripts/local-top/live" \
    "$MOUNT/usr/share/initramfs-tools/hooks/live" \
    "$MOUNT/usr/share/initramfs-tools/conf-hooks.d/live" \
    "$MOUNT/etc/initramfs-tools/conf.d/live" \
    "$MOUNT/etc/initramfs-tools/conf.d/resume" \
    "$MOUNT/etc/live" \
    "$MOUNT/lib/live" \
    "$MOUNT/etc/systemd/system/live-config.service" \
    "$MOUNT/lib/systemd/system/live-config.service"; do
    rm -rf "$live_path" 2>/dev/null || true
done

find "$MOUNT/usr/share/initramfs-tools/" -name "*live*" -delete 2>/dev/null || true
find "$MOUNT/etc/initramfs-tools/"       -name "*live*" -delete 2>/dev/null || true

rm -f "$MOUNT/boot/"initrd.img* \
      "$MOUNT/boot/"System.map* \
      "$MOUNT/boot/"config-* 2>/dev/null || true

echo "50"
for d in dev dev/pts proc sys run; do
    mkdir -p "$MOUNT/$d"
done

mount --bind /dev     "$MOUNT/dev"
mount --bind /dev/pts "$MOUNT/dev/pts"
mount --bind /proc    "$MOUNT/proc"
mount --bind /sys     "$MOUNT/sys"
mount --bind /run     "$MOUNT/run"

if [ "$UEFI" -eq 1 ] && [ -d /sys/firmware/efi ]; then
    mkdir -p "$MOUNT/sys/firmware/efi/efivars"
    mount --bind /sys/firmware/efi/efivars "$MOUNT/sys/firmware/efi/efivars"
fi

cp -L /etc/resolv.conf "$MOUNT/etc/resolv.conf" || true

echo "Sistem ayarlanıyor..."

KERNEL_VER="$(uname -r)"
[ -n "$KERNEL_VER" ] || die "Kernel sürümü bulunamadı!"

if [ -f "/live/vmlinuz" ]; then
    cp -f /live/vmlinuz "$MOUNT/boot/vmlinuz-$KERNEL_VER"
elif [ -f "/run/live/medium/live/vmlinuz" ]; then
    cp -f /run/live/medium/live/vmlinuz "$MOUNT/boot/vmlinuz-$KERNEL_VER"
elif [ -f "/boot/vmlinuz-$KERNEL_VER" ]; then
    cp -f "/boot/vmlinuz-$KERNEL_VER" "$MOUNT/boot/vmlinuz-$KERNEL_VER"
else
    die "Canlı sistem kernel dosyası bulunamadı!"
fi

chroot "$MOUNT" useradd -m -s /bin/bash -G sudo,audio,video,plugdev,netdev "$USERNAME" || true
echo "$USERNAME:$SIFRE" | chroot "$MOUNT" chpasswd

echo "$HOSTNAME" > "$MOUNT/etc/hostname"
cat > "$MOUNT/etc/hosts" <<EOF
127.0.0.1 localhost
127.0.1.1 $HOSTNAME
::1 localhost ip6-localhost ip6-loopback
EOF

ROOT_UUID=$(blkid -s UUID -o value "$P2")

if [ "$UEFI" -eq 1 ]; then
    EFI_UUID=$(blkid -s UUID -o value "$P1")
    cat > "$MOUNT/etc/fstab" <<EOF
UUID=$ROOT_UUID  /         ext4  defaults,errors=remount-ro  0 1
UUID=$EFI_UUID   /boot/efi vfat  defaults,umask=0077         0 1
EOF
else
    cat > "$MOUNT/etc/fstab" <<EOF
UUID=$ROOT_UUID  /  ext4  defaults,errors=remount-ro  0 1
EOF
fi

echo "Display manager'lar devre dışı bırakılıyor..."
for dm in sddm lightdm gdm gdm3 lxdm xdm; do
    chroot "$MOUNT" systemctl disable "${dm}.service" 2>/dev/null || true
    chroot "$MOUNT" systemctl mask    "${dm}.service" 2>/dev/null || true
done

echo "TTY1 otomatik giriş ayarlanıyor: $USERNAME"
mkdir -p "$MOUNT/etc/systemd/system/getty@tty1.service.d"
cat > "$MOUNT/etc/systemd/system/getty@tty1.service.d/override.conf" <<EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $USERNAME --noclear %I \$TERM
Type=idle
EOF
chroot "$MOUNT" systemctl enable getty@tty1

echo "Kullanıcı shell profilleri yapılandırılıyor..."
USERHOME="$MOUNT/home/$USERNAME"

for profile_file in .bash_profile .profile .bashrc; do
    cat > "$USERHOME/$profile_file" <<EOF
if [[ -z \$DISPLAY ]] && [[ \$(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF
done

mkdir -p "$MOUNT/etc/profile.d"
cat > "$MOUNT/etc/profile.d/autostartx.sh" <<EOF
#!/bin/sh
if [ "\$(id -u)" != "0" ] && [ -z "\$DISPLAY" ] && [ "\$(tty)" = "/dev/tty1" ]; then
    exec startx
fi
EOF
chmod +x "$MOUNT/etc/profile.d/autostartx.sh"

echo ".xinitrc yapılandırılıyor..."
cat > "$USERHOME/.xinitrc" <<'EOF'
export XDG_SESSION_TYPE=x11
export LANG=tr_TR.UTF-8
export LC_ALL=tr_TR.UTF-8
export LC_MESSAGES=tr_TR.UTF-8
export LANGUAGE=tr_TR:tr
xhost +local:
gio set "$HOME/Desktop/pusula-finans.desktop" metadata::trust true 2>/dev/null || true
gio set "$HOME/Desktop/crystal-setup.desktop" metadata::trust true 2>/dev/null || true
gio set "$HOME/Desktop/pusula-ai.desktop" metadata::trust true 2>/dev/null || true
pulseaudio --start &
xrdb -merge "$HOME/.Xresources"
setxkbmap tr
exec startlxqt
EOF
rm -f /home/$USERNAME/Desktop/crystal-setup.desktop 2>/dev/null || true

echo "Klavye düzeni yapılandırılıyor..."
cat > "$MOUNT/etc/default/keyboard" <<EOF
XKBMODEL=""
XKBLAYOUT="tr"
XKBVARIANT=""
XKBOPTIONS=""
BACKSPACE="guess"
EOF
mkdir -p "$MOUNT/etc/X11/xorg.conf.d"
cat > "$MOUNT/etc/X11/xorg.conf.d/00-keyboard.conf" <<EOF
Section "InputClass"
    Identifier "system-keyboard"
    MatchIsKeyboard "on"
    Option "XkbLayout"  "tr"
    Option "XkbVariant" ""
    Option "XkbOptions" ""
EndSection
EOF
chroot "$MOUNT" setupcon --save 2>/dev/null || true
chroot "$MOUNT" dpkg-reconfigure -f noninteractive keyboard-configuration 2>/dev/null || true

###
echo "PCManFM-Qt masaüstü ayarları yapılandırılıyor..."
mkdir -p "$USERHOME/Desktop"
mkdir -p "$USERHOME/.config/pcmanfm-qt/lxqt"
cat > "$USERHOME/.config/pcmanfm-qt/lxqt/settings.conf" <<EOF
[Desktop]
DesktopShortcuts=Trash, Computer
Wallpaper=/usr/share/backgrounds/crystalos.png
WallpaperMode=stretch
[System]
Terminal=qterminal
TerminalDirCommand=qterminal -w %s
TerminalExecCommand=qterminal -e %s
EOF
cat > "$MOUNT/etc/xdg/pcmanfm-qt/lxqt/settings.conf" <<'EOF'
[System]
Terminal=qterminal
TerminalDirCommand=qterminal -w %s
TerminalExecCommand=qterminal -e %s
EOF
cat > "$MOUNT/etc/xdg/autostart/nm-tray.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=nm-tray
Exec=nm-tray
OnlyShowIn=LXQt;
Terminal=false
EOF
chroot "$MOUNT" update-alternatives --install \
  /usr/bin/x-terminal-emulator x-terminal-emulator /usr/bin/qterminal 50 || true
chroot "$MOUNT" update-alternatives --set \
  x-terminal-emulator /usr/bin/qterminal || true
cat > "$USERHOME/.config/qterminal.ini" <<EOF
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
mkdir -p "$USERHOME/.config/lxqt"
cat > "$USERHOME/.config/lxqt/lxqt.conf" <<EOF
[General]
icon_theme=Papirus
EOF
cat > "$USERHOME/.config/lxqt/panel.conf" <<EOF
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
mkdir -p "$USERHOME/.config/lxqt-panel"
chroot "$MOUNT" chown -R "$USERNAME:$USERNAME" "/home/$USERNAME"
chroot "$MOUNT" systemctl enable NetworkManager
mkdir -p "$MOUNT/etc/gtk-3.0"
cat > "$MOUNT/etc/gtk-3.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF
mkdir -p "$MOUNT/etc/gtk-4.0"
cat > "$MOUNT/etc/gtk-4.0/settings.ini" <<EOF
[Settings]
gtk-theme-name = Numix
gtk-icon-theme-name = Papirus
EOF
mkdir -p "$MOUNT/etc/gtk-2.0"
cat > "$MOUNT/etc/gtk-2.0/gtkrc" <<EOF
gtk-theme-name="Numix"
gtk-icon-theme-name="Papirus"
EOF
mkdir -p "$USERHOME/.config/openbox"
cp "$SCRIPT_DIR/rc.xml" "$USERHOME/.config/openbox/rc.xml"
cat > "$USERHOME/.Xresources" <<EOF
Xft.dpi: 110
Xft.antialias: 1
Xft.hinting: 1
Xft.hintstyle: hintslight
Xft.rgba: rgb
EOF
mkdir -p "$MOUNT/etc/xdg/lxqt"
cat > "$MOUNT/etc/xdg/lxqt/session.conf" <<EOF
[General]
window_manager=openbox
EOF
cat > "$MOUNT/etc/profile.d/qt-platformtheme.sh" <<'EOF'
#!/bin/sh
export QT_QPA_PLATFORMTHEME=qt5ct
EOF
chmod +x "$MOUNT/etc/profile.d/qt-platformtheme.sh"
mkdir -p "$USERHOME/.config/qt5ct"
cat > "$USERHOME/.config/qt5ct/qt5ct.conf" <<EOF
[Appearance]
style=gtk2
icon_theme=Papirus
font=Sans,10,-1,5,50,0,0,0,0,0
EOF
mkdir -p "$USERHOME/.config/qt6ct"
cat > "$USERHOME/.config/qt6ct/qt6ct.conf" <<EOF
[Appearance]
style=gtk2
icon_theme=Papirus
font=Sans,10,-1,5,50,0,0,0,0,0
EOF
chroot "$MOUNT" chown -R "$USERNAME:$USERNAME" /opt/pusula-finans
mkdir -p "$MOUNT/usr/share/applications"
cat > "$MOUNT/usr/share/applications/pusula-finans.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Pusula Finans
Exec=python3 /opt/pusula-finans/pusula_finans.py
Icon=/opt/pusula-finans/img/logo1.png
Terminal=false
Categories=Finance;
EOF
chroot "$MOUNT" chown "$USERNAME:$USERNAME" /usr/share/applications/pusula-finans.desktop
cp "$MOUNT/usr/share/applications/pusula-finans.desktop" "$USERHOME/Desktop/pusula-finans.desktop"
cat > "$MOUNT/usr/share/applications/pusula-ai.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Pusula AI
Path=/opt/Pusula-AI/v1.5
Exec=python3 /opt/Pusula-AI/v1.5/pusula-ai.py
Icon=/opt/Pusula-AI/img/logo.png
Terminal=true
Categories=Education;
EOF
cp "$MOUNT/usr/share/applications/pusula-ai.desktop" \
    "$USERHOME/Desktop/pusula-ai.desktop"
chmod +x "$USERHOME/Desktop/pusula-ai.desktop"
cat > "$MOUNT/usr/share/applications/vlc.desktop" <<'EOF'
[Desktop Entry]
Version=1.0
Name=VLC media player
GenericName=Media player
Comment=Read, capture, broadcast your multimedia streams
Name[tr]=VLC ortam oynatıcısı
GenericName[tr]=Ortam oynatıcısı
Comment[tr]=Çoklu ortam akışlarınızı okuyun, yakalayın, yayınlayın
Exec=vlc %U
TryExec=/usr/bin/vlc
Icon=vlc
Terminal=false
Type=Application
Categories=AudioVideo;Player;Recorder;
MimeType=application/ogg;application/x-ogg;audio/ogg;audio/vorbis;audio/x-vorbis;audio/x-vorbis+ogg;video/ogg;video/x-ogm;video/x-ogm+ogg;video/x-theora+ogg;video/x-theora;audio/x-speex;audio/opus;application/x-flac;audio/flac;audio/x-flac;audio/x-ms-asf;audio/x-ms-asx;audio/x-ms-wax;audio/x-ms-wma;video/x-ms-asf;video/x-ms-asf-plugin;video/x-ms-asx;video/x-ms-wm;video/x-ms-wmv;video/x-ms-wmx;video/x-ms-wvx;video/x-msvideo;audio/x-pn-windows-acm;video/divx;video/msvideo;video/vnd.divx;video/avi;video/x-avi;application/vnd.rn-realmedia;application/vnd.rn-realmedia-vbr;audio/vnd.rn-realaudio;audio/x-pn-realaudio;audio/x-pn-realaudio-plugin;audio/x-real-audio;audio/x-realaudio;video/vnd.rn-realvideo;audio/mpeg;audio/mpg;audio/mp1;audio/mp2;audio/mp3;audio/x-mp1;audio/x-mp2;audio/x-mp3;audio/x-mpeg;audio/x-mpg;video/mp2t;video/mpeg;video/mpeg-system;video/x-mpeg;video/x-mpeg2;video/x-mpeg-system;application/mpeg4-iod;application/mpeg4-muxcodetable;application/x-extension-m4a;application/x-extension-mp4;audio/aac;audio/m4a;audio/mp4;audio/x-m4a;audio/x-aac;video/mp4;video/mp4v-es;video/x-m4v;application/x-quicktime-media-link;application/x-quicktimeplayer;video/quicktime;application/x-matroska;audio/x-matroska;video/x-matroska;video/webm;audio/webm;audio/3gpp;audio/3gpp2;audio/AMR;audio/AMR-WB;video/3gp;video/3gpp;video/3gpp2;x-scheme-handler/mms;x-scheme-handler/mmsh;x-scheme-handler/rtsp;x-scheme-handler/rtp;x-scheme-handler/rtmp;x-scheme-handler/icy;x-scheme-handler/icyx;application/x-cd-image;x-content/video-vcd;x-content/video-svcd;x-content/video-dvd;x-content/audio-cdda;x-content/audio-player;application/ram;application/xspf+xml;audio/mpegurl;audio/x-mpegurl;audio/scpls;audio/x-scpls;text/google-video-pointer;text/x-google-video-pointer;video/vnd.mpegurl;application/vnd.apple.mpegurl;application/vnd.ms-asf;application/vnd.ms-wpl;application/sdp;audio/dv;video/dv;audio/x-aiff;audio/x-pn-aiff;video/x-anim;video/x-nsv;video/fli;video/flv;video/x-flc;video/x-fli;video/x-flv;audio/wav;audio/x-pn-au;audio/x-pn-wav;audio/x-wav;audio/x-adpcm;audio/ac3;audio/eac3;audio/vnd.dts;audio/vnd.dts.hd;audio/vnd.dolby.heaac.1;audio/vnd.dolby.heaac.2;audio/vnd.dolby.mlp;audio/basic;audio/midi;audio/x-ape;audio/x-gsm;audio/x-musepack;audio/x-tta;audio/x-wavpack;audio/x-shorten;application/x-shockwave-flash;application/x-flash-video;misc/ultravox;image/vnd.rn-realpix;audio/x-it;audio/x-mod;audio/x-s3m;audio/x-xm;application/mxf;
X-KDE-Protocols=ftp,http,https,mms,rtmp,rtsp,sftp,smb
Keywords=Player;Capture;DVD;Audio;Video;Server;Broadcast;
EOF
chroot "$MOUNT" chown "$USERNAME:$USERNAME" /usr/share/applications/vlc.desktop
DESKTOP="$MOUNT/usr/share/applications/org.kde.falkon.desktop"
if [ -f "$DESKTOP" ]; then
    sed 's|^Exec=env QTWEBENGINE_DISABLE_SANDBOX=1 falkon %u|Exec=falkon %u|' \
        "$DESKTOP" > "${DESKTOP}.tmp" && \
    mv "${DESKTOP}.tmp" "$DESKTOP"
fi
###

mkdir -p "$MOUNT/etc/initramfs-tools"
cat > "$MOUNT/etc/initramfs-tools/initramfs.conf" <<'EOF'
MODULES=most
BUSYBOX=y
COMPRESS=gzip
BOOT=local
RESUME=none
EOF

cat > "$MOUNT/etc/initramfs-tools/modules" <<'EOF'
ext4
vfat
fat
amdgpu
i915
drm
drm_kms_helper
EOF

mkdir -p "$MOUNT/etc/initramfs-tools/hooks"
cat > "$MOUNT/etc/initramfs-tools/hooks/gpu-firmware" <<'HOOK'
#!/bin/sh
PREREQS=""
prereqs() { echo "$PREREQS"; }
case "$1" in prereqs) prereqs; exit 0;; esac
. /usr/share/initramfs-tools/hook-functions

# AMD APU/GPU firmware
if [ -d /lib/firmware/amdgpu ]; then
    cp -r /lib/firmware/amdgpu "${DESTDIR}/lib/firmware/" 2>/dev/null || true
fi
# Intel GPU firmware
if [ -d /lib/firmware/i915 ]; then
    cp -r /lib/firmware/i915 "${DESTDIR}/lib/firmware/" 2>/dev/null || true
fi
HOOK
chmod +x "$MOUNT/etc/initramfs-tools/hooks/gpu-firmware"

echo "İnitramfs yeniden oluşturuluyor..."
echo "60"
chroot "$MOUNT" depmod -a "$KERNEL_VER" || true
rm -f "$MOUNT/boot/initrd.img-$KERNEL_VER" 2>/dev/null || true
chroot "$MOUNT" update-initramfs -c -k "$KERNEL_VER"

if chroot "$MOUNT" lsinitramfs "/boot/initrd.img-$KERNEL_VER" 2>/dev/null | grep -qE '(^|/)scripts/live'; then
    echo "UYARI: initramfs içinde hâlâ live kalıntısı var, son temizlik yapılıyor..."
    find "$MOUNT/usr/share/initramfs-tools/" -name "*live*" -delete 2>/dev/null || true
    chroot "$MOUNT" update-initramfs -d -k "$KERNEL_VER" 2>/dev/null || true
    chroot "$MOUNT" update-initramfs -c -k "$KERNEL_VER"
fi

ln -sf "vmlinuz-$KERNEL_VER"    "$MOUNT/boot/vmlinuz"
ln -sf "initrd.img-$KERNEL_VER" "$MOUNT/boot/initrd.img"

echo "GRUB kuruluyor..."
echo "70"
if [ "$UEFI" -eq 1 ]; then
    chroot "$MOUNT" grub-install \
        --target=x86_64-efi \
        --efi-directory=/boot/efi \
        --bootloader-id=CrystalOS \
        --recheck
else
    chroot "$MOUNT" grub-install --target=i386-pc "$DISK"
fi

mkdir -p "$MOUNT/boot/grub"
cat > "$MOUNT/boot/grub/grub.cfg" <<EOF
set timeout=5
set default=0

menuentry "CrystalOS" {
    linux  /boot/vmlinuz-$KERNEL_VER root=UUID=$ROOT_UUID ro quiet splash
    initrd /boot/initrd.img-$KERNEL_VER
}
menuentry "CrystalOS (guvenli mod)" {
    linux  /boot/vmlinuz-$KERNEL_VER root=UUID=$ROOT_UUID ro debug
    initrd /boot/initrd.img-$KERNEL_VER
}
EOF

echo "95"
sync || true
cleanup_mounts
echo "100"
) | "${YAD_BASE[@]}" --progress --title="CrystalOS Kuruluyor" \
    --text="Kurulum devam ediyor, lütfen bekleyin..." --percentage=0 \
    --width=450 --auto-close --no-cancel

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    "${YAD_BASE[@]}" --question --title="Kurulum Tamamlandı" \
        --text="CrystalOS başarıyla kuruldu!\n\nYeniden başlatmak istiyor musunuz?" \
        --button="Yeniden Başlat:0" --button="Kapat:1" && reboot
else
    "${YAD_BASE[@]}" --error --title="Kurulum Başarısız" \
        --text="Kurulum sırasında bir hata oluştu.\nAyrıntılar için $LOGFILE dosyasına bakın."
fi
