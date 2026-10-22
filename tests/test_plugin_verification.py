"""Regression tests for plugin verification, independent of an LMMS install."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from lmms_mcp import lmms_app, server
from lmms_mcp.project import LMMSProject


class PluginVerificationTests(unittest.TestCase):
    def setUp(self):
        # Never probe or execute the workstation's application during unit tests.
        self.exe = patch.object(lmms_app, "find_lmms_exe", return_value=Path("/test/lmms"))
        self.version = patch.object(lmms_app, "get_lmms_version", return_value="1.2.2")
        self.exe.start()
        self.version.start()
        self.addCleanup(self.exe.stop)
        self.addCleanup(self.version.stop)
        self.previous_project = server.get_project()
        project = LMMSProject()
        project.new()
        server.set_project(project)
        self.addCleanup(server.set_project, self.previous_project)

    def test_empty_inventory_does_not_verify_any_plugins(self):
        with patch.object(lmms_app, "get_installed_plugins", return_value=set()):
            info = json.loads(server.get_lmms_info())
        self.assertEqual(info["instruments_verified_available"], [])
        self.assertEqual(info["effects_verified_available"], [])

    def test_missing_application_is_not_verified(self):
        with patch.object(lmms_app, "find_lmms_exe", return_value=None):
            info = json.loads(server.get_lmms_info())
        self.assertFalse(info["found"])
        self.assertEqual(info.get("instruments_verified_available", []), [])

    def test_detected_plugin_and_alias_remain_verified(self):
        inventory = {"tripleoscillator", "papu", "delay"}
        with patch.object(lmms_app, "get_installed_plugins", return_value=inventory):
            info = json.loads(server.get_lmms_info())
        self.assertIn("tripleoscillator", info["instruments_verified_available"])
        self.assertIn("nes", info["instruments_verified_available"])
        self.assertIn("delay", info["effects_verified_available"])
        self.assertNotIn("slicert", info["instruments_verified_available"])

    def test_static_plugin_assumption_is_not_verification(self):
        with patch.object(lmms_app, "get_installed_plugins", return_value={"tripleoscillator"}):
            info = json.loads(server.get_lmms_info())
        self.assertNotIn("freeboy", info["instruments_verified_available"])

    def test_offline_composition_still_adds_an_instrument(self):
        with patch.object(lmms_app, "get_installed_plugins", return_value=set()):
            result = json.loads(server.add_instrument_track("Offline lead"))
        self.assertNotIn("error", result)
        self.assertEqual(len(server.get_project().get_info().tracks), 1)


if __name__ == "__main__":
    unittest.main()
