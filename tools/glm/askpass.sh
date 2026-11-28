#!/bin/sh
# Пароль SSH к серверу GLM для ssh (SSH_ASKPASS): берётся из .env (PASS_SSH_PASS), в файлах репозитория его нет.
grep '^PASS_SSH_PASS=' "${DAMP_ENV:-.env}" | head -1 | cut -d= -f2- | tr -d '\r\n'
