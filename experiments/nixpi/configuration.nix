# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, lib, pkgs, ... }:

let
  go-librespot = pkgs.buildGoModule rec {
    pname = "go-librespot";
    version = "b4a9dd9";

    src = pkgs.fetchFromGitHub {
      owner = "devgianlu";
      repo = "go-librespot";
      rev = "b4a9dd9c4f6b13502008122cb787f3c3a683d8d0";
      hash = "sha256-42WkTBsDbXHmvpt7jb6A4s1Tnchj1j5yUFqxZ30f/24=";
    };

    vendorHash = "sha256-vjJ7jt8kzCYVfDeQQmfOe32PaKqKv+gc/rMMNPMvYt4=";

    env.CGO_ENABLED = 1;
    # Stop nix builds complaining about missing events package.
    postPatch = "mkdir -p events/impl && echo package impl > events/impl/impl.go";

    nativeBuildInputs = [ pkgs.pkg-config ];
    buildInputs = [ pkgs.libogg pkgs.libvorbis pkgs.alsa-lib ];

    subPackages = [ "./cmd/daemon" ];

    meta = {
      description = "Yet another open source Spotify client, written in Go.";
      homepage = "https://github.com/devgianlu/go-librespot";
      license = lib.licenses.gpl3Only;
      maintainers = with lib.maintainers; [ ];
    };
  };
in

{
  imports =
    [ # Include the results of the hardware scan.
      ./hardware-configuration.nix
    ];

  # Use the extlinux boot loader. (NixOS wants to enable GRUB by default)
  boot.loader.grub.enable = false;
  # Enables the generation of /boot/extlinux/extlinux.conf
  boot.loader.generic-extlinux-compatible.enable = true;

  networking.hostName = "speaker";

  # Configure network connections interactively with nmcli or nmtui.
  networking.networkmanager.enable = true;

  # Set your time zone.
  time.timeZone = "Europe/Amsterdam";

  # Enable Broadcom Audio driver
  boot.kernelParams = [ "snd_bcm2835.enable_hdmi=1" "snd_bcm2835.enable_headphones=1" ];

  # Enable touchpad support (enabled default in most desktopManager).
  # services.libinput.enable = true;

  # Define a user account. Don't forget to set a password with ‘passwd’.
  users.users.daan = {
    isNormalUser = true;
    extraGroups = [ "wheel" "audio" ]; # Group wheel enables sudo
  };

  # List packages installed in system profile.
  environment.systemPackages = with pkgs; [
    vim
    go-librespot
    tmux
  ];

  # List services that you want to enable:

  # Enable the OpenSSH daemon.
  services.openssh.enable = true;

  # Go-librespot
  systemd.services.go-librespot = {
     wantedBy = [ "multi-user.target" ];
     wants = [ "network-online.target" ];
     after = [ "network-online.target" "audio.target" ];
     description = "Go-librespot Spotify daemon";
     serviceConfig = {
       Type = "simple";
       User = "daan"; # Configuration read from home dir
       SupplementaryGroups = "audio";
       ExecStart = ''${go-librespot}/bin/daemon'';
       Restart="always";
       RestartSec=12;
     };
  };

  # Disable the firewall.
  # TODO: Re-enable but allow the necessary ports
  networking.firewall.enable = false;
  # networking.firewall.allowedTCPPorts = [ ... ];
  # networking.firewall.allowedUDPPorts = [ ... ];

  # This option defines the first version of NixOS you have installed on this particular machine,
  # and is used to maintain compatibility with application data (e.g. databases) created on older NixOS versions.
  #
  # Most users should NEVER change this value after the initial install, for any reason,
  # even if you've upgraded your system to a new NixOS release.
  #
  # This value does NOT affect the Nixpkgs version your packages and OS are pulled from,
  # so changing it will NOT upgrade your system - see https://nixos.org/manual/nixos/stable/#sec-upgrading for how
  # to actually do that.
  #
  # This value being lower than the current NixOS release does NOT mean your system is
  # out of date, out of support, or vulnerable.
  #
  # Do NOT change this value unless you have manually inspected all the changes it would make to your configuration,
  # and migrated your data accordingly.
  #
  # For more information, see `man configuration.nix` or https://nixos.org/manual/nixos/stable/options#opt-system.stateVersion .
  system.stateVersion = "25.11"; # Did you read the comment?
}
