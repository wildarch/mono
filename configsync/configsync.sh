#!/usr/bin/env bash

# Backs up and restores configuration and secrets to Google Drive.

# Exit on error
set -e
# Exit if a piped command fails
set -o pipefail

if [[ -z "$RCLONE" ]]; then
    echo "env RCLONE must be set (are you running this script using bazel?)"
    exit 1
fi

function rclone_config_create() {
    $RCLONE config create configbackup drive \
        scope drive.appfolder \
        root_folder_id appDataFolder
}

function rclone_backup_ssh() {
    $RCLONE sync "$HOME"/.ssh configbackup:ssh
    echo "Backed up $HOME/.ssh"
}

function rclone_backup_failmail() {
    file=/etc/failmail.conf
    sudo cat $file | $RCLONE rcat configbackup:failmail.conf
    echo "Backed up $file"
}

function rclone_backup_gitconfig() {
    file="$HOME/.gitconfig"
    $RCLONE copy $file configbackup:
    echo "Backed up $file"
}

function rclone_restore_ssh() {
    $RCLONE copy configbackup:/ssh ~/.ssh/
    for file in ~/.ssh/id_*; do
        echo "$file"
        if [[ $file == *.pub ]]; then
            chmod 644 "$file"
        else
            chmod 600 "$file"
        fi
    done
    echo "Restored SSH Keys"
}

function rclone_restore_failmail() {
    $RCLONE copy configbackup:failmail.conf /tmp/
    sudo install --mode=600 --owner=root --group=root \
        /tmp/failmail.conf /etc/failmail.conf
    rm -f /tmp/failmail.conf
    echo "Restored failmail config"
}

function rclone_restore_gitconfig() {
    $RCLONE copy configbackup:.gitconfig "$HOME"
    echo "Restored git config"
}

case $1 in
    backup) 
        rclone_config_create
        rclone_backup_ssh
        rclone_backup_failmail
        rclone_backup_gitconfig
        ;;
    restore)
        rclone_config_create
        rclone_restore_ssh
        rclone_restore_failmail
        rclone_restore_gitconfig
        ;;
    *)
        echo $"Usage: $0 {backup|restore}"
        exit 1
esac