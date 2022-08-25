#!/usr/bin/env bash

systemctl daemon-reload
systemctl enable cast-backend.service
systemctl restart cast-backend.service