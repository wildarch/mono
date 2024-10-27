#!/usr/bin/env bash

systemctl stop cast-backend.service
systemctl disable cast-backend.service
systemctl daemon-reload