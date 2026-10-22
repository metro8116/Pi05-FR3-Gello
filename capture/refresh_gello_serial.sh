#!/usr/bin/env bash
# Refresh GELLO device links at startup; never open or write to the serial port.
set -Eeuo pipefail
export LC_ALL=C
export DOCKER_HOST="${LOCAL_ROS2_DOCKER_HOST:-unix:///var/run/docker.sock}"

fail() { printf '[gello-serial] %s\n' "$*" >&2; exit 1; }
[[ $# == 2 && "$1" == --container ]] || fail "Usage: $0 --container CONTAINER_ID"
container_id="$2"
[[ "$container_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "Invalid container identifier"

ports=()
if [[ -n "${GELLO_PORT:-}" ]]; then
    [[ "$GELLO_PORT" == /dev/serial/by-id/* ]] || fail "GELLO_PORT must be a stable /dev/serial/by-id path"
    ports+=("$GELLO_PORT")
else
    while IFS= read -r port; do
        case "${port##*/}" in
            usb-FTDI_USB__-__Serial_Converter*|usb-ROBOTIS_OpenRB-150*) ports+=("$port") ;;
        esac
    done < <(find /dev/serial/by-id -maxdepth 1 -type l -print 2>/dev/null)
fi
[[ ${#ports[@]} == 1 ]] || fail "Expected exactly one host GELLO FTDI/OpenRB device; found ${#ports[@]}"
port_id="${ports[0]##*/}"
tty_path="$(readlink -f -- "${ports[0]}")" || fail "Cannot resolve host GELLO serial link"
[[ "$tty_path" =~ ^/dev/tty(USB|ACM)[0-9]+$ ]] || fail "Unexpected GELLO device path: $tty_path"
device_stat="$(stat -Lc '%F|%t|%T' -- "$tty_path")" || fail "Cannot stat host device $tty_path"
IFS='|' read -r device_type major_hex minor_hex <<< "$device_stat"
[[ "$device_type" == 'character special file' && "$major_hex" =~ ^[a-fA-F0-9]+$ && "$minor_hex" =~ ^[a-fA-F0-9]+$ ]] || fail "Host GELLO target is not a valid character device"
major=$((16#$major_hex))
minor=$((16#$minor_hex))

docker exec -i "$container_id" /bin/bash -s -- "$port_id" "$tty_path" "$major" "$minor" <<'CONTAINER'
set -Eeuo pipefail
export LC_ALL=C
fail() { printf '[gello-serial] %s\n' "$*" >&2; exit 1; }
port_id="$1"; node="$2"; major="$3"; minor="$4"
backing=/tmp/gello-serial-by-id
by_id=/dev/serial/by-id
expected="$(printf '%x:%x' "$major" "$minor")"
if [[ -e "$node" || -L "$node" ]]; then
    actual="$(stat -Lc '%F|%t:%T' -- "$node")" || fail "Cannot stat existing container device $node"
    [[ "$actual" == "character special file|$expected" ]] || fail "Refusing to replace existing container device $node: $actual, expected $expected"
else
    mknod -m 0660 "$node" c "$major" "$minor" || fail "Cannot create missing container device $node"
fi
mkdir -p "$backing"
for old in "$backing"/usb-FTDI_USB__-__Serial_Converter* "$backing"/usb-ROBOTIS_OpenRB-150*; do
    [[ -L "$old" ]] && rm -- "$old"
done
ln -sfn -- "$node" "$backing/$port_id"
if [[ "$(readlink -f -- "$by_id/$port_id" 2>/dev/null || true)" != "$node" ]]; then
    if mountpoint -q "$by_id"; then
        umount -l "$by_id" || fail "Cannot detach stale $by_id mount"
    fi
    mkdir -p "$by_id" || fail "Cannot recreate $by_id"
    mount --bind "$backing" "$by_id" || fail "Cannot bind GELLO serial links into $by_id"
fi
[[ "$(readlink -f -- "$by_id/$port_id")" == "$node" ]] || fail "GELLO serial link does not resolve to $node"
[[ "$(stat -Lc '%F|%t:%T' -- "$by_id/$port_id")" == "character special file|$expected" ]] || fail "GELLO serial device number verification failed"
printf '[gello-serial] %s -> %s (major:minor %s:%s)\n' "$by_id/$port_id" "$node" "$major" "$minor"
CONTAINER
