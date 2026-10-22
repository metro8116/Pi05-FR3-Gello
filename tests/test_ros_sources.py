"""Source preparation checks use tiny synthetic archives and mocked HTTPS."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import ssl
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_sources", ROOT / "docker" / "prepare_sources.py")
sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sources)


def archive_bytes(name="source/readme.txt", text=b"pinned source", kind=None, link=""):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind is not None:
            member.type, member.linkname = kind, link
        else:
            member.size = len(text)
        archive.addfile(member, io.BytesIO(text) if member.isfile() else None)
    return output.getvalue()


def item_for(data):
    commit = "a" * 40
    return dict(name="fixture", repository="frankarobotics/fixture", commit=commit,
                filename="fixture.tar.gz", url="https://codeload.github.com/frankarobotics/fixture/tar.gz/" + commit,
                sha256=hashlib.sha256(data).hexdigest(), accepted_sha256=[],
                archive_roots=["source"], max_bytes=1024 * 1024)


class Response(io.BytesIO):
    def __init__(self, data, url, headers=None):
        super().__init__(data)
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        value = self.payloads.pop(0)
        if isinstance(value, Exception):
            raise value
        return Response(value, request.full_url)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.vendor = self.root / "vendor"
        self.data = archive_bytes()
        self.item = item_for(self.data)

    def tearDown(self):
        self.directory.cleanup()

    def test_compatible_with_host_python38_grammar(self):
        ast.parse((ROOT / "docker/prepare_sources.py").read_text(), feature_version=8)

    def test_committed_lock_has_four_sources_and_exact_common_gitlink(self):
        items = sources.load_lock(ROOT / "docker/sources.lock.json")
        indexed = {item["name"]: item for item in items}
        self.assertEqual(len(items), 4)
        self.assertEqual(indexed["libfranka"]["submodules"]["common"], indexed["libfranka-common"]["commit"])

    def test_verified_vendor_is_used_without_creating_network_opener(self):
        self.vendor.mkdir()
        destination = self.vendor / self.item["filename"]
        destination.write_bytes(self.data)
        with patch.object(sources, "_opener", side_effect=AssertionError("network opened")):
            state, result = sources.prepare_source(self.item, self.vendor)
        self.assertEqual((state, result), ("cached", destination))

    def test_explicit_cache_is_imported_offline_and_original_is_unchanged(self):
        cache = self.root / "explicit_cache"
        cache.mkdir()
        candidate = cache / self.item["filename"]
        candidate.write_bytes(self.data)
        state, result = sources.prepare_source(self.item, self.vendor, [cache], offline=True)
        self.assertEqual(state, "imported")
        self.assertEqual(result.read_bytes(), self.data)
        self.assertEqual(candidate.read_bytes(), self.data)
        self.assertEqual(list(self.vendor.iterdir()), [result])

    def test_audited_alternate_archive_hash_is_accepted(self):
        self.vendor.mkdir()
        alternate = archive_bytes(text=b"same source representation with local metadata")
        self.item["accepted_sha256"] = [hashlib.sha256(alternate).hexdigest()]
        destination = self.vendor / self.item["filename"]
        destination.write_bytes(alternate)
        self.assertEqual(sources.prepare_source(self.item, self.vendor, offline=True)[0], "cached")

    def test_corrupt_existing_vendor_is_preserved_and_never_downloaded_over(self):
        self.vendor.mkdir()
        destination = self.vendor / self.item["filename"]
        destination.write_bytes(b"damaged archive")
        with patch.object(sources, "_opener", side_effect=AssertionError("network opened")):
            with self.assertRaisesRegex(sources.SourceError, "SHA256 mismatch"):
                sources.prepare_source(self.item, self.vendor)
        self.assertEqual(destination.read_bytes(), b"damaged archive")

    def test_bad_explicit_cache_is_never_copied(self):
        cache = self.root / "explicit_cache"
        cache.mkdir()
        (cache / self.item["filename"]).write_bytes(b"wrong hash")
        with self.assertRaisesRegex(sources.SourceError, "offline.*SHA256 mismatch"):
            sources.prepare_source(self.item, self.vendor, [cache], offline=True)
        self.assertEqual(list(self.vendor.iterdir()), [])

    def test_missing_offline_source_does_not_open_network(self):
        with patch.object(sources, "_opener", side_effect=AssertionError("network opened")):
            with self.assertRaisesRegex(sources.SourceError, "offline mode"):
                sources.prepare_source(self.item, self.vendor, offline=True)

    def test_failed_hash_is_retried_and_temporary_files_are_removed(self):
        opener, delays = Opener(b"wrong hash", self.data), []
        state, result = sources.prepare_source(self.item, self.vendor, retries=2, opener=opener, sleep=delays.append)
        self.assertEqual(state, "downloaded")
        self.assertEqual(delays, [1])
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(list(self.vendor.iterdir()), [result])
        self.assertEqual(result.read_bytes(), self.data)

    def test_network_failures_do_not_publish_partial_archives(self):
        opener = Opener(OSError("TLS connection interrupted"), OSError("still interrupted"))
        with self.assertRaisesRegex(sources.SourceError, "after 2 attempts"):
            sources.prepare_source(self.item, self.vendor, retries=2, opener=opener, sleep=lambda _: None)
        self.assertEqual(list(self.vendor.iterdir()), [])

    def test_atomic_publication_never_overwrites_a_concurrent_valid_winner(self):
        alternate = archive_bytes(text=b"concurrent audited winner")
        self.item["accepted_sha256"] = [hashlib.sha256(alternate).hexdigest()]
        destination = self.vendor / self.item["filename"]

        def competing_publish(*args, **kwargs):
            destination.write_bytes(alternate)
            raise FileExistsError("concurrent writer won")

        with patch.object(sources.os, "link", side_effect=competing_publish):
            sources.prepare_source(self.item, self.vendor, retries=1, opener=Opener(self.data))
        self.assertEqual(destination.read_bytes(), alternate)
        self.assertEqual(list(self.vendor.iterdir()), [destination])

    def test_size_and_total_time_limits(self):
        self.item["max_bytes"] = len(self.data) - 1
        with self.assertRaisesRegex(sources.SourceError, "size limit"):
            sources.prepare_source(self.item, self.vendor, retries=1, opener=Opener(self.data))
        self.assertEqual(list(self.vendor.iterdir()), [])
        with tempfile.TemporaryFile() as output:
            with self.assertRaisesRegex(sources.SourceError, "total time limit"):
                sources._copy_bounded(io.BytesIO(self.data), output, self.item, time.monotonic() - 1)

    def test_archive_traversal_links_and_devices_are_rejected_even_with_matching_hash(self):
        cases = [("source/../../escape", None, ""), ("/source/absolute", None, ""),
                 ("other/file", None, ""), ("source/link", tarfile.SYMTYPE, "../../escape"),
                 ("source/link", tarfile.LNKTYPE, "../escape"), ("source/device", tarfile.CHRTYPE, "")]
        for name, kind, link in cases:
            with self.subTest(name=name, kind=kind):
                payload = archive_bytes(name=name, kind=kind, link=link)
                item = item_for(payload)
                destination = self.root / "unsafe.tar.gz"
                destination.write_bytes(payload)
                with self.assertRaisesRegex(sources.SourceError, "Unsafe"):
                    sources.verify_archive(destination, item)

    def test_symlink_vendor_and_archive_are_rejected(self):
        target = self.root / "target"
        target.mkdir()
        self.vendor.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(sources.SourceError, "symlink"):
            sources.prepare_source(self.item, self.vendor, offline=True)
        self.vendor.unlink()
        self.vendor.mkdir()
        archive = self.root / "real_archive.tar.gz"
        archive.write_bytes(self.data)
        (self.vendor / self.item["filename"]).symlink_to(archive)
        with self.assertRaises(sources.SourceError):
            sources.prepare_source(self.item, self.vendor, offline=True)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO support")
    def test_nonregular_archive_cannot_block_waiting_for_a_fifo_writer(self):
        self.vendor.mkdir()
        destination = self.vendor / self.item["filename"]
        os.mkfifo(str(destination))
        with self.assertRaisesRegex(sources.SourceError, "regular file"):
            sources.prepare_source(self.item, self.vendor, offline=True)

    def test_tls_certificate_verification_and_redirect_host_are_enforced(self):
        opener = sources._opener()
        handler = next(value for value in opener.handlers if hasattr(value, "_context"))
        self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handler._context.check_hostname)
        redirect = sources.HTTPSRedirectOnly()
        for url in ["http://codeload.github.com/unsafe", "https://example.com/archive"]:
            with self.assertRaises(sources.SourceError):
                redirect.redirect_request(None, None, 302, "redirect", {}, url)

    def test_lock_rejects_paths_non_https_sources_and_unpinned_urls(self):
        for key, value in [("filename", "../archive.tar.gz"), ("url", "http://codeload.github.com/archive"),
                           ("commit", "main"), ("url", self.item["url"][:-40] + "main")]:
            item = dict(self.item)
            item[key] = value
            lock = self.root / "lock.json"
            lock.write_text(json.dumps(dict(schema_version=1, sources=[item])))
            with self.assertRaises(sources.SourceError):
                sources.load_lock(lock)


if __name__ == "__main__":
    unittest.main()
