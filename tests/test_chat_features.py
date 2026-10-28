"""Chat pictures, transcript storage, and manual-mode prompt text."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from app.agent import _system_extra
from app.chat_images import ChatImageError, image_file, save_chat_images
from app.memory import MemoryStore
from app.llm import grid_to_page, rewrite_grid_coordinates
from app.manual import is_command_request, is_screen_question, latest_screenshot
from app.skills import SkillError, add_skill_request, replace_skill, save_task_as_skill

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


class _Action:
    def __init__(self, action: str, details: str) -> None:
        self.action = action
        self.details = details


class _Task:
    def __init__(self, task: str, result: str, actions: list[_Action]) -> None:
        self.task = task
        self.result = result
        self.actions = actions


def _details(action: dict[str, object], goal: str, url: str = "", reason: str = "") -> str:
    payload: dict[str, object] = {"target": json.dumps(action), "goal": goal, "url": url}
    if reason:
        payload["reason"] = reason
    return json.dumps(payload)


class ManualModeTests(unittest.TestCase):
    def test_questions_are_not_commands(self) -> None:
        self.assertTrue(is_command_request("commands"))
        self.assertTrue(is_screen_question("Tell me the coordinate of the first blue bottle."))
        self.assertTrue(is_screen_question("Where is the refresh button?"))
        self.assertFalse(is_screen_question("click 346 556"))
        self.assertFalse(is_screen_question("right click 420 310"))
        self.assertFalse(is_screen_question("press Escape"))

    def test_latest_screenshot_is_the_newest_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "task-1"
            newer = root / "task-2"
            older.mkdir()
            newer.mkdir()
            first = older / "step-001.png"
            second = newer / "step-002.png"
            first.write_bytes(b"old")
            second.write_bytes(b"new")
            os_utime = __import__("os").utime
            os_utime(first, (1, 1))
            os_utime(second, (2, 2))
            self.assertEqual(latest_screenshot(root), second)

    def test_model_grid_becomes_page_pixels(self) -> None:
        # The purple fish sat at pixel (528, 389) on a 1248x671 page.
        # The model reported that point as 423, 580 on its 0-1000 grid.
        self.assertEqual(grid_to_page(423, 580, 1248, 671), (528, 389))
        reply = rewrite_grid_coordinates(
            "The purple fish is located at coordinate_x=423, coordinate_y=580.",
            1248,
            671,
        )
        self.assertIn("coordinate_x=528, coordinate_y=389", reply)
        self.assertIn("click 528 389", reply)


class AddSkillTests(unittest.TestCase):
    def test_command_and_saved_skill(self) -> None:
        self.assertEqual(add_skill_request("Add skill"), "")
        self.assertEqual(add_skill_request("Add skill price-check"), "price-check")
        self.assertIsNone(add_skill_request("Open Google"))
        address = "0x006FaD88c35D973A87E451CF8D000c7e83Dad409"
        task = (
            f"Open https://etherscan.io/address/{address}#code\n"
            "Read the contract.\n\n"
            "Verified Solidity source is already saved on disk.\n"
            f"Saved file: contracts/{address}/Token.sol"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = save_task_as_skill(
                [(2, "Add skill"), (1, task)],
                directory=root,
            )
            self.assertEqual(skill.args, ("address",))
            self.assertEqual(skill.fetch, "solidity")
            self.assertIn("{address}", skill.body)
            self.assertNotIn("Saved file:", skill.body)
            rendered = (root / skill.path.name).read_text(encoding="utf-8")
            self.assertIn("validate: address=ethereum", rendered)
            with self.assertRaises(SkillError):
                save_task_as_skill([(1, "Add skill")], directory=root)

    def test_skill_uses_completed_steps_and_updates_the_same_file(self) -> None:
        refused = _Task(
            "Buy a new windows server in bill.ultahost.com.",
            "Unable to complete this request. Blocked automatically: financial transactions.",
            [_Action("start", _details({}, "start"))],
        )
        worked = _Task(
            "Buy a new windows server in Ultahost.",
            "Stopped before checkout.",
            [
                _Action("navigate", _details({"url": "https://ultahost.com"}, "Navigate to ultahost.com")),
                _Action("click", _details({"index": 2}, "Click on Hosting in the navigation menu.")),
                _Action(
                    "scroll",
                    _details({"down": True}, "Scroll down to see the plans, then select a plan to purchase."),
                ),
                _Action("click", _details({"coordinate_x": 510, "coordinate_y": 310}, "Click the captcha animal.")),
                _Action(
                    "blocked",
                    _details({"index": 9}, "Click Order Now.", reason="Blocked automatically: financial transactions."),
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_task_as_skill([refused, worked], directory=root)
            second = save_task_as_skill([refused, worked], directory=root)
            self.assertEqual(first.path, second.path)
            self.assertEqual(list(root.glob("*.md")), [first.path])
            self.assertIn("Open https://ultahost.com.", first.body)
            self.assertIn("Click on Hosting in the navigation menu.", first.body)
            self.assertNotIn("510", first.body)
            self.assertNotIn("captcha", first.body.lower())
            self.assertIn("Stop before payment or checkout.", first.body)
            self.assertNotIn("plan to purchase", first.body)
            self.assertIn("Scroll down to see the plans.", first.body)

    def test_replace_skill_keeps_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_task_as_skill([(1, "Open the billing page and stop.")], directory=root)
            updated = replace_skill(
                "billing-page-stop",
                name="billing-page",
                description="Open billing",
                args=[],
                validate="",
                fetch="",
                body="Open the billing page.\nStop on the invoice list.",
                directory=root,
            )
            self.assertEqual(updated.name, "billing-page")
            self.assertIn("invoice list", updated.body)
            self.assertFalse((root / "billing-page-stop.md").exists())
            self.assertTrue((root / "billing-page.md").exists())


if __name__ == "__main__":
    unittest.main()
