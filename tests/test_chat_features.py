"""Chat pictures, transcript storage, and manual-mode prompt text."""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from app.agent import _system_extra
from app.chat_images import ChatImageError, image_file, save_chat_images
from app.memory import MemoryStore

_PNG = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
).decode("ascii")


class ChatFeatureTests(unittest.IsolatedAsyncioTestCase):
    def test_save_and_reject_pictures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urls, kept = save_chat_images([f"data:image/png;base64,{_PNG}"], root)
            self.assertEqual(len(urls), 1)
            self.assertTrue(urls[0].startswith("/chat/images/"))
            self.assertTrue(kept[0].startswith("data:image/png;base64,"))
            name = urls[0].rsplit("/", 1)[1]
            self.assertIsNotNone(image_file(root, name))
            self.assertIsNone(image_file(root, "../agent.db"))
            with self.assertRaises(ChatImageError):
                save_chat_images(["data:image/svg+xml;base64,PHN2Zy8+"], root)

    def test_manual_prompt_drops_standing_instructions(self) -> None:
        manual = _system_extra(True)
        auto = _system_extra(False)
        self.assertIn("Manual mode.", manual)
        self.assertIn("Standing instructions do not apply.", manual)
        self.assertIn("Standing instructions from a similar past situation also apply.", auto)

    async def test_chat_stores_picture_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = MemoryStore(root / "agent.db", root / "actions.jsonl")
            await memory.init()
            await memory.add_chat("user", "print(1)", images=["/chat/images/abc.png"])
            await memory.add_chat("user", "", images=["/chat/images/def.png"])
            lines = await memory.list_chat()
            self.assertEqual(lines[0].text, "print(1)")
            self.assertEqual(lines[0].images, ("/chat/images/abc.png",))
            self.assertEqual(lines[1].images, ("/chat/images/def.png",))


if __name__ == "__main__":
    unittest.main()
