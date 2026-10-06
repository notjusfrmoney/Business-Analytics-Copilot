from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.errors import StreamlitSecretNotFoundError

import src.utils.config as config


class StreamlitSecretsConfigTests(unittest.TestCase):
    def test_streamlit_secrets_supply_openrouter_settings(self) -> None:
        secrets = {
            "OPENROUTER_API_KEY": "secret-key",
            "OPENROUTER_MODEL": "openrouter/free",
            "OPENROUTER_BASE_URL": "https://example.test/v1/",
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(config, "st", SimpleNamespace(secrets=secrets)),
        ):
            settings = config.get_openrouter_settings()

        self.assertEqual(
            settings,
            ("secret-key", "openrouter/free", "https://example.test/v1"),
        )

    def test_environment_values_take_precedence_over_streamlit_secrets(self) -> None:
        environment = {
            "OPENROUTER_API_KEY": "environment-key",
            "OPENROUTER_MODEL": "environment/model",
        }
        secrets = {
            "OPENROUTER_API_KEY": "secret-key",
            "OPENROUTER_MODEL": "secret/model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(config, "st", SimpleNamespace(secrets=secrets)),
        ):
            settings = config.get_openrouter_settings()

        self.assertEqual(settings[:2], ("environment-key", "environment/model"))

    def test_missing_streamlit_secrets_are_safe_outside_streamlit(self) -> None:
        class MissingSecrets:
            def __getitem__(self, key: str) -> str:
                raise StreamlitSecretNotFoundError("No secrets file.")

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(config, "st", SimpleNamespace(secrets=MissingSecrets())),
        ):
            values = config.load_project_config()
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                config.get_openrouter_settings()

        self.assertEqual(values["OPENROUTER_API_KEY"], "")
        self.assertEqual(values["OPENROUTER_MODEL"], "")
        self.assertEqual(
            values["OPENROUTER_BASE_URL"],
            "https://openrouter.ai/api/v1",
        )


if __name__ == "__main__":
    unittest.main()
