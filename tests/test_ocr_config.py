import unittest

from app.ocr_config import resolve_ocr_config


class ResolveOcrConfigTest(unittest.TestCase):
    def test_remote_without_valid_token_falls_back_to_local(self):
        backend, token, fallback_message = resolve_ocr_config(
            selected_backend="remote",
            token=None,
            local_backend="local",
            remote_backend="remote",
            token_length=40,
        )

        self.assertEqual(backend, "local")
        self.assertIsNone(token)
        self.assertIn("本地", fallback_message)

    def test_remote_with_valid_token_stays_remote(self):
        backend, token, fallback_message = resolve_ocr_config(
            selected_backend="remote",
            token="x" * 40,
            local_backend="local",
            remote_backend="remote",
            token_length=40,
        )

        self.assertEqual(backend, "remote")
        self.assertEqual(token, "x" * 40)
        self.assertIsNone(fallback_message)


if __name__ == "__main__":
    unittest.main()
