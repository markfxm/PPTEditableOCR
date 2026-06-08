import unittest

from app.ocr_config import resolve_ocr_config, should_clear_paused_ocr_request


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

    def test_new_load_clears_paused_request_but_resume_does_not(self):
        self.assertTrue(should_clear_paused_ocr_request(force_local_ocr=False))
        self.assertFalse(should_clear_paused_ocr_request(force_local_ocr=True))


if __name__ == "__main__":
    unittest.main()
