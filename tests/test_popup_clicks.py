"""Clicks land on the popup in front, not the page behind it."""

from __future__ import annotations

import logging
import unittest

from app.popup_clicks import (
    _matching_frame_id,
    click_visible_layer,
    iframe_local_point,
    install_popup_clicks,
    is_popup_center_click,
    pick_control,
    right_click_point,
    should_click_front_layer,
)


class Rect:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class InputApi:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def dispatchMouseEvent(self, params: dict, session_id: str | None = None) -> dict:
        self.session.events.append(params)
        return {}


class DomApi:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def enable(self, session_id: str | None = None) -> dict:
        return {}

    async def getNodeForLocation(self, params: dict, session_id: str | None = None) -> dict:
        return {"backendNodeId": 1}

    async def describeNode(self, params: dict, session_id: str | None = None) -> dict:
        node: dict = {"nodeName": "IFRAME"}
        if self.session.child_frame_id:
            node["frameId"] = self.session.child_frame_id
        return {"node": node}


class RuntimeApi:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def evaluate(self, params: dict, session_id: str | None = None) -> dict:
        return {"result": {"value": self.session.hit}}


class Send:
    def __init__(self, session: Session) -> None:
        self.DOM = DomApi(session)
        self.Runtime = RuntimeApi(session)
        self.Input = InputApi(session)


class Client:
    def __init__(self, session: Session) -> None:
        self.send = Send(session)


class Session:
    def __init__(self, name: str, hit: dict, child_frame_id: str | None = None) -> None:
        self.name = name
        self.session_id = name
        self.hit = hit
        self.child_frame_id = child_frame_id
        self.events: list[dict] = []
        self.cdp_client = Client(self)


class Browser:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self.sessions = sessions
        self.frames: dict[str, dict] = {}

    async def cdp_client_for_node(self, node: object) -> Session:
        return self.sessions["page"]

    async def cdp_client_for_frame(self, frame_id: str) -> Session:
        return self.sessions[frame_id]

    async def get_element_coordinates(self, backend_node_id: int, session: Session) -> Rect:
        return Rect(100, 50, 200, 100)

    async def get_or_create_cdp_session(self) -> Session:
        return self.sessions["page"]

    async def get_all_frames(self) -> tuple[dict[str, dict], dict]:
        return self.frames, {}


class Node:
    def __init__(self, tag: str) -> None:
        self.tag_name = tag
        self.attributes: dict[str, str] = {}
        self.backend_node_id = 7


class Watchdog:
    def __init__(self, browser: Browser) -> None:
        self.browser_session = browser
        self.logger = logging.getLogger("popup-test")
        self.occlusion = False

    async def _check_element_occlusion(
        self, backend_node_id: int, x: float, y: float, session: Session
    ) -> bool:
        return self.occlusion


class PopupClickTests(unittest.IsolatedAsyncioTestCase):
    def test_geometry_and_matching(self) -> None:
        self.assertEqual(iframe_local_point(630, 578, 400, 200, 2, 2), (228, 376))
        self.assertTrue(should_click_front_layer("IFRAME", False))
        self.assertTrue(should_click_front_layer("BUTTON", True))
        self.assertFalse(should_click_front_layer("BUTTON", False))
        self.assertTrue(is_popup_center_click("iframe", 400, 300))
        self.assertFalse(is_popup_center_click("button", 200, 40, covers_frame=True))
        self.assertTrue(
            is_popup_center_click(
                "canvas",
                380,
                260,
                inside_frame=True,
                frame_width=400,
                frame_height=300,
            )
        )
        skip = {"left": 800, "top": 610, "right": 870, "bottom": 650, "text": "skip"}
        move = {"left": 700, "top": 280, "right": 760, "bottom": 320, "text": "Move"}
        self.assertEqual(pick_control(859, 631, [move, skip]), (835, 630))
        self.assertEqual(pick_control(400, 400, [move, skip]), None)
        frames = {
            "short": {"url": "https://"},
            "checkbox": {"url": "https://newassets.hcaptcha.com/captcha/v1/checkbox"},
            "challenge": {"url": "https://newassets.hcaptcha.com/captcha/v1/challenge.html"},
        }
        self.assertEqual(
            _matching_frame_id(frames, "https://newassets.hcaptcha.com/captcha/v1/challenge.html"),
            "challenge",
        )

    async def test_click_enters_the_popup_frame(self) -> None:
        page = Session("page", {"tag": "IFRAME", "frame": True, "localX": 40, "localY": 70, "src": ""}, "popup")
        popup = Session("popup", {"tag": "CANVAS", "frame": False})
        browser = Browser({"page": page, "popup": popup})

        result = await click_visible_layer(Watchdog(browser), page, 200, 100)

        self.assertEqual(result, {"click_x": 40.0, "click_y": 70.0})
        self.assertEqual(page.events, [])
        self.assertEqual([event["type"] for event in popup.events], ["mouseMoved", "mousePressed", "mouseReleased"])
        self.assertEqual(popup.events[-1]["x"], 40.0)
        self.assertEqual(popup.events[-1]["y"], 70.0)
        self.assertEqual(popup.events[1]["button"], "left")

    async def test_right_click_uses_the_right_button(self) -> None:
        page = Session("page", {"tag": "BUTTON", "frame": False})
        browser = Browser({"page": page})

        await right_click_point(browser, 120, 80)

        pressed = page.events[1]
        self.assertEqual(pressed["type"], "mousePressed")
        self.assertEqual(pressed["button"], "right")
        self.assertEqual(pressed["buttons"], 2)

    async def test_element_click_uses_the_popup_when_it_covers_the_page(self) -> None:
        from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

        install_popup_clicks()
        page = Session(
            "page",
            {
                "tag": "IFRAME",
                "frame": True,
                "localX": 15,
                "localY": 25,
                "src": "https://newassets.hcaptcha.com/captcha/v1/challenge.html",
            },
            None,
        )
        popup = Session("popup", {"tag": "CANVAS", "frame": False})
        browser = Browser({"page": page, "popup": popup})
        browser.frames = {
            "popup": {"url": "https://newassets.hcaptcha.com/captcha/v1/challenge.html"},
        }
        watchdog = Watchdog(browser)

        result = await DefaultActionWatchdog._click_element_node_impl(watchdog, Node("button"))

        self.assertEqual(result, {"click_x": 15.0, "click_y": 25.0})
        self.assertEqual(page.events, [])
        self.assertTrue(popup.events)

    async def test_clicking_the_popup_frame_does_not_use_its_center(self) -> None:
        from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

        install_popup_clicks()
        page = Session("page", {"tag": "IFRAME", "frame": True, "localX": 200, "localY": 150, "src": ""})
        browser = Browser({"page": page})
        watchdog = Watchdog(browser)

        result = await DefaultActionWatchdog._click_element_node_impl(watchdog, Node("iframe"))

        self.assertIn("coordinate_x", result["validation_error"])
        self.assertEqual(page.events, [])

    async def test_covered_element_is_not_activated_through_the_popup(self) -> None:
        from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

        install_popup_clicks()
        page = Session("page", {"tag": "DIV", "frame": False})
        browser = Browser({"page": page})
        watchdog = Watchdog(browser)
        watchdog.occlusion = True

        result = await DefaultActionWatchdog._click_element_node_impl(watchdog, Node("button"))

        self.assertEqual(result["click_x"], 200.0)
        self.assertEqual(result["click_y"], 100.0)
        self.assertEqual([event["type"] for event in page.events], ["mouseMoved", "mousePressed", "mouseReleased"])

    async def test_uncovered_element_is_left_to_the_normal_click(self) -> None:
        from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

        install_popup_clicks()
        page = Session("page", {"tag": "BUTTON", "frame": False})
        browser = Browser({"page": page})
        watchdog = Watchdog(browser)
        watchdog.occlusion = False

        with self.assertRaises(Exception):
            await DefaultActionWatchdog._click_element_node_impl(watchdog, Node("button"))

        self.assertEqual(page.events, [])


if __name__ == "__main__":
    unittest.main()
