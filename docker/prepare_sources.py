#!/usr/bin/env python3
"""Prepare hash-locked ROS source archives without git or third-party modules.

Compatible with the Ubuntu 20.04 host's Python 3.8. Existing vendor archives
are never replaced. Only explicitly supplied cache directories are searched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import ssl
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
CHUNK_BYTES = 1024 * 1024
MAX_MEMBERS = 100000
MAX_UNPACKED_BYTES = 2 * 1024 ** 3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class SourceError(RuntimeError):
    pass


def load_lock(path):
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1 or not isinstance(document.get("sources"), list):
        raise SourceError("Unsupported ROS source lock")
    names, filenames = set(), set()
    for item in document["sources"]:
        if not isinstance(item, dict):
            raise SourceError("Invalid source lock entry")
        name, filename = item.get("name"), item.get("filename")
        commit, repository = item.get("commit"), item.get("repository")
        hashes = [item.get("sha256")] + item.get("accepted_sha256", [])
        roots = item.get("archive_roots")
        if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name)
                or not isinstance(filename, str) or not SAFE_NAME.fullmatch(filename)
                or not filename.endswith(".tar.gz") or filename in filenames or name in names
                or not isinstance(commit, str) or not COMMIT.fullmatch(commit)
                or not isinstance(repository, str)
                or not re.fullmatch(r"frankarobotics/[A-Za-z0-9_.-]+", repository)
                or not hashes or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes)
                or not isinstance(roots, list) or not roots
                or any(not isinstance(value, str) or not SAFE_NAME.fullmatch(value) for value in roots)
                or type(item.get("max_bytes")) is not int or not 0 < item["max_bytes"] <= 512 * 1024 ** 2
                or item.get("url") != "https://codeload.github.com/{}/tar.gz/{}".format(repository, commit)):
            raise SourceError("Invalid source lock entry: {}".format(name))
        names.add(name)
        filenames.add(filename)
    if not names:
        raise SourceError("ROS source lock must not be empty")
    return document["sources"]


def _open_regular(path):
    try:
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise SourceError("Source archive must be a regular file: {}".format(path))
        return os.fdopen(descriptor, "rb")
    except OSError as exc:
        raise SourceError("Cannot safely open source archive {}: {}".format(path, exc)) from exc


def _check_archive(path, item):
    """Inspect members without extracting; reject traversal and special files."""
    count, unpacked = 0, 0
    try:
        with _open_regular(path) as data, tarfile.open(fileobj=data, mode="r:gz") as archive:
            for member in archive:
                count += 1
                unpacked += member.size
                parts = PurePosixPath(member.name).parts
                if (count > MAX_MEMBERS or unpacked > MAX_UNPACKED_BYTES
                        or member.size < 0 or not parts or member.name.startswith("/")
                        or "\\" in member.name or ".." in parts
                        or parts[0] not in item["archive_roots"]
                        or not (member.isfile() or member.isdir() or member.issym() or member.islnk())):
                    raise SourceError("Unsafe or oversized tar member in {}: {}".format(path, member.name))
                if member.issym() or member.islnk():
                    link = member.linkname
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), link)
                                                if member.issym() else link)
                    if (not link or link.startswith("/") or "\\" in link
                            or target.split("/")[0] != parts[0]):
                        raise SourceError("Unsafe tar link in {}: {}".format(path, member.name))
        if count == 0:
            raise SourceError("Empty source archive: {}".format(path))
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise SourceError("Invalid source archive {}: {}".format(path, exc)) from exc


def verify_archive(path, item):
    digest, size = hashlib.sha256(), 0
    with _open_regular(path) as stream:
        while True:
            data = stream.read(CHUNK_BYTES)
            if not data:
                break
            size += len(data)
            if size > item["max_bytes"]:
                raise SourceError("Source archive exceeds size limit: {}".format(path))
            digest.update(data)
    actual = digest.hexdigest()
    accepted = [item["sha256"]] + item.get("accepted_sha256", [])
    if actual not in accepted:
        raise SourceError("SHA256 mismatch for {}: got {}".format(path, actual))
    _check_archive(path, item)
    return actual


class HTTPSRedirectOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.hostname != "codeload.github.com":
            raise SourceError("Refusing source download redirect outside HTTPS codeload.github.com")
        return super().redirect_request(request, response, code, message, headers, newurl)


def _opener():
    # The default context verifies the certificate chain and hostname.
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()), HTTPSRedirectOnly())


def _copy_bounded(source, destination, item, deadline):
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise SourceError("Source preparation exceeded total time limit")
        block = source.read(CHUNK_BYTES)
        if not block:
            break
        size += len(block)
        if size > item["max_bytes"]:
            raise SourceError("Source archive exceeds size limit for {}".format(item["name"]))
        destination.write(block)
    destination.flush()
    os.fsync(destination.fileno())


def _publish(temporary, destination, item):
    verify_archive(temporary, item)
    os.chmod(str(temporary), 0o644)
    try:
        # Same-filesystem hard linking is atomic and cannot overwrite a winner
        # from another preparation process, unlike os.replace/rename.
        os.link(str(temporary), str(destination), follow_symlinks=False)
    except FileExistsError:
        verify_archive(destination, item)
    return destination


def prepare_source(item, vendor, caches=(), offline=False, retries=3,
                   timeout=30.0, total_timeout=180.0, opener=None, sleep=time.sleep):
    vendor = Path(vendor)
    if vendor.is_symlink():
        raise SourceError("Vendor directory must not be a symlink: {}".format(vendor))
    vendor.mkdir(parents=True, exist_ok=True)
    destination = vendor / item["filename"]
    if destination.exists() or destination.is_symlink():
        # A corrupt existing file is intentionally preserved for investigation.
        verify_archive(destination, item)
        return "cached", destination
    errors = []
    for directory in caches:
        candidate = Path(directory).expanduser() / item["filename"]
        if not candidate.exists() and not candidate.is_symlink():
            continue
        temporary = None
        try:
            verify_archive(candidate, item)
            descriptor, name = tempfile.mkstemp(prefix="." + item["filename"] + ".tmp-", dir=str(vendor))
            temporary = Path(name)
            with _open_regular(candidate) as source, os.fdopen(descriptor, "wb") as output:
                _copy_bounded(source, output, item, time.monotonic() + total_timeout)
            return "imported", _publish(temporary, destination, item)
        except (OSError, SourceError) as exc:
            errors.append(str(exc))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if offline:
        raise SourceError("Missing verified archive {} in offline mode{}".format(
            destination, "; " + "; ".join(errors) if errors else ""))
    if opener is None:
        opener = _opener()
    for attempt in range(retries):
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix="." + item["filename"] + ".tmp-", dir=str(vendor))
            temporary = Path(name)
            request = urllib.request.Request(item["url"], headers={"User-Agent": "Pi05-FR3-source-preparer/1"})
            with os.fdopen(descriptor, "wb") as output, opener.open(request, timeout=timeout) as response:
                final = urllib.parse.urlsplit(response.geturl())
                if final.scheme != "https" or final.hostname != "codeload.github.com":
                    raise SourceError("Source download must remain on HTTPS codeload.github.com")
                length = response.headers.get("Content-Length")
                if length and (not length.isdigit() or int(length) > item["max_bytes"]):
                    raise SourceError("Invalid or oversized source Content-Length")
                _copy_bounded(response, output, item, time.monotonic() + total_timeout)
            return "downloaded", _publish(temporary, destination, item)
        except (OSError, ValueError, urllib.error.URLError, SourceError) as exc:
            errors.append(str(exc))
            if attempt + 1 < retries:
                sleep(min(2 ** attempt, 4))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise SourceError("Could not prepare {} after {} attempts: {}".format(
        item["name"], retries, "; ".join(errors[-retries:])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--vendor-dir", type=Path, default=ROOT / "vendor")
    parser.add_argument("--cache-dir", type=Path, action="append", default=[],
                        help="Explicit existing archive directory to import; repeatable")
    parser.add_argument("--offline", action="store_true", help="Use verified vendor/cache files only")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30., help="Socket timeout, seconds")
    parser.add_argument("--total-timeout", type=float, default=180., help="Copy time budget per attempt, seconds")
    args = parser.parse_args(argv)
    if not 1 <= args.retries <= 5 or not 0 < args.timeout <= 120 or not 0 < args.total_timeout <= 600:
        parser.error("retries must be 1..5, timeout in (0,120], total-timeout in (0,600]")
    try:
        sources = load_lock(args.lock)
        for item in sources:
            state, path = prepare_source(item, args.vendor_dir, args.cache_dir, args.offline,
                                         args.retries, args.timeout, args.total_timeout)
            print("[ros-sources] {} {}".format(state, path), flush=True)
    except (OSError, ValueError, TypeError, SourceError) as exc:
        print("[ros-sources] ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
