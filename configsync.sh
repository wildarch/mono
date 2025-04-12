#!/usr/bin/env bash

# Backs up and restores configuration and secrets to Google Drive.

# Exit on error
set -e
# Exit if a piped command fails
set -o pipefail

function rclone_config_create() {
    rclone config create configbackup drive \
        scope drive.appfolder \
        root_folder_id appDataFolder
}

function rclone_backup_ssh() {
    rclone sync "$HOME"/.ssh configbackup:ssh
    echo "Backed up $HOME/.ssh"
}

function rclone_backup_gitconfig() {
    file="$HOME/.gitconfig"
    rclone copy "$file" configbackup:
    echo "Backed up $file"
}

function rclone_restore_ssh() {
    rclone copy configbackup:/ssh ~/.ssh/
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

function rclone_restore_gitconfig() {
    rclone copy configbackup:.gitconfig "$HOME"
    echo "Restored git config"
}

case $1 in
    backup) 
        rclone_config_create
        rclone_backup_ssh
        rclone_backup_gitconfig
        ;;
    restore)
        rclone_config_create
        rclone_restore_ssh
        rclone_restore_gitconfig
        ;;
    *)
        echo $"Usage: $0 {backup|restore}"
        exit 1
esac