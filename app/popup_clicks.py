"""Deliver clicks to the popup in front of the page.

Browser Use treats a covered control as still clickable and fires
``element.click()`` on it. That click reaches the page behind a popup.
A real pointer event is sent to whatever is painted on top, and when that
layer is a frame it is delivered inside the frame.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("ai_browser_agent")

_MAX_FRAME_DEPTH = 5
_INSTALLED = False

_POINT_EXPRESSION = """
(function(x, y) {
  function frameInfo(el) {
    const rect = el.getBoundingClientRect();
    const width = Math.max(rect.width, 0);
    const height = Math.max(rect.height, 0);
    const rawX = x - rect.left - (el.clientLeft || 0);
    const rawY = y - rect.top - (el.clientTop || 0);
    return {
      tag: 'IFRAME',
      frame: true,
      src: el.src || '',
      width: width,
      height: height,
      localX: Math.min(Math.max(rawX, 0), Math.max(width - 1, 0)),
      localY: Math.min(Math.max(rawY, 0), Math.max(height - 1, 0))
    };
  }
  function interactive(el) {
    if (!el || !el.closest) return false;
    return !!el.closest('button, a, input, select, textarea, [role="button"], [role="link"]');
  }
  const stack = document.elementsFromPoint(x, y) || [];
  for (const el of stack) {
    const tag = String(el.tagName || '').toUpperCase();
    if (tag === 'IFRAME' || tag === 'FRAME') return frameInfo(el);
    if (interactive(el)) break;
  }
  let nearest = null;
  let nearestDist = 24;
  for (const el of document.querySelectorAll('iframe, frame')) {
    const rect = el.getBoundingClientRect();
    if (rect.width < 80 || rect.height < 80) continue;
    const clampedX = Math.min(Math.max(x, rect.left), rect.right);
    const clampedY = Math.min(Math.max(y, rect.top), rect.bottom);
    const dist = Math.hypot(x - clampedX, y - clampedY);
    if (dist <= nearestDist) {
      nearestDist = dist;
      nearest = el;
    }
  }
  if (nearest) return frameInfo(nearest);
  const top = stack[0];
  if (!top) return null;
  return { tag: String(top.tagName || '').toUpperCase(), frame: false };
})(%s, %s)
"""

_CONTROLS_EXPRESSION = """
(function(x, y) {
  function box(el) {
    const rect = el.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      text: String(el.innerText || el.getAttribute('aria-label') || '').slice(0, 40)
    };
  }
  function isControl(el) {
    if (!el || el.nodeType !== 1) return false;
    const tag = el.tagName;
    if (tag === 'BUTTON' || tag === 'A' || tag === 'INPUT') return true;
    const role = el.getAttribute('role');
    if (role === 'button' || role === 'link') return true;
    if (/button/i.test(String(el.className || ''))) {
      const rect = el.getBoundingClientRect();
      return rect.width >= 16 && rect.width <= 240 && rect.height >= 16 && rect.height <= 90;
    }
    return false;
  }
  const controls = [];
  const seen = new Set();
  function add(el) {
    if (!el || seen.has(el) || !isControl(el)) return;
    seen.add(el);
    controls.push(box(el));
  }
  for (const el of document.elementsFromPoint(x, y) || []) {
    let node = el;
    while (node && node !== document.documentElement) {
      if (isControl(node)) {
        add(node);
        break;
      }
      node = node.parentElement;
    }
  }
  for (const el of document.querySelectorAll('button, a, [role="button"], [role="link"]')) {
    const rect = el.getBoundingClientRect();
    const dx = Math.max(rect.left - x, 0, x - rect.right);
    const dy = Math.max(rect.top - y, 0, y - rect.bottom);
    if (Math.hypot(dx, dy) <= 16) add(el);
  }
  return { controls: controls };
})(%s, %s)
"""


def iframe_local_point(
    page_x: float,
    page_y: float,
    left: float,
    top: float,
    client_left: float = 0,
    client_top: float = 0,
) -> tuple[float, float]:
    """Convert a page coordinate into the coordinate space of a frame."""

    return (
        max(0.0, page_x - left - client_left),
        max(0.0, page_y - top - client_top),
    )


def pick_control(
    x: float,
    y: float,
    controls: list[dict[str, Any]],
    margin: float = 16,
) -> tuple[float, float] | None:
    """Return the center of the button at this point, or the nearest one within ``margin``."""

    nearest: dict[str, Any] | None = None
    nearest_center: tuple[float, float] | None = None
    nearest_dist = margin
    for control in controls:
        try:
            left = float(control["left"])
            top = float(control["top"])
            right = float(control["right"])
            bottom = float(control["bottom"])
        except (KeyError, TypeError, ValueError):
            continue
        if right <= left or bottom <= top:
            continue
        if left <= x <= right and top <= y <= bottom:
            return (left + right) / 2, (top + bottom) / 2
        dx = max(left - x, 0, x - right)
        dy = max(top - y, 0, y - bottom)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= nearest_dist:
            nearest = control
            nearest_dist = dist
            nearest_center = ((left + right) / 2, (top + bottom) / 2)
    if nearest is None:
        return None
    return nearest_center


def should_click_front_layer(top_tag: str, occluded: bool) -> bool:
    """True when the painted layer is a popup covering the chosen element."""

    return top_tag in {"IFRAME", "FRAME"} or occluded


def is_popup_center_click(
    tag: str,
    element_width: float,
    element_height: float,
    *,
    covers_frame: bool = False,
    inside_frame: bool = False,
    frame_width: float | None = None,
    frame_height: float | None = None,
) -> bool:
    """True when the click target is the popup itself, so its center is not a target."""

    if tag.lower() in {"iframe", "frame"}:
        return True
    if not covers_frame and not inside_frame:
        return False
    if frame_width and frame_height and frame_width > 0 and frame_height > 0:
        return element_width >= frame_width * 0.7 and element_height >= frame_height * 0.7
    return tag.lower() in {"img", "canvas"} and element_width >= 180 and element_height >= 140


_POPUP_CENTER_MESSAGE = (
    "This click hits the center of the popup. "
    "Click the target's point inside the popup with coordinate_x and coordinate_y."
)


def install_popup_clicks() -> None:
    """Route Browser Use clicks to the front popup. Safe to call more than once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    original_click = DefaultActionWatchdog._click_element_node_impl
    original_coordinate = DefaultActionWatchdog._click_on_coordinate

    async def _click_element_node_impl(self: Any, element_node: Any) -> dict | None:
        tag = (getattr(element_node, "tag_name", None) or "").lower()
        attributes = getattr(element_node, "attributes", None) or {}
        element_type = str(attributes.get("type", "")).lower()
        if tag == "select" or (tag == "input" and element_type == "file"):
            return await original_click(self, element_node)

        try:
            session = await self.browser_session.cdp_client_for_node(element_node)
            rect = await self.browser_session.get_element_coordinates(
                element_node.backend_node_id, session
            )
            if rect is not None and rect.width > 0 and rect.height > 0:
                x = rect.x + rect.width / 2
                y = rect.y + rect.height / 2
                info = await _point_info(session, x, y)
                top = str((info or {}).get("tag") or "")
                frame_box = info if info and info.get("frame") else None
                if is_popup_center_click(
                    tag,
                    rect.width,
                    rect.height,
                    covers_frame=frame_box is not None,
                    inside_frame=_inside_frame(self.browser_session, session),
                    frame_width=_positive(frame_box, "width"),
                    frame_height=_positive(frame_box, "height"),
                ):
                    self.logger.info("Skipping the center of the popup.")
                    return {"validation_error": _POPUP_CENTER_MESSAGE}
                occluded = False
                if top not in {"IFRAME", "FRAME"}:
                    occluded = await self._check_element_occlusion(
                        element_node.backend_node_id, x, y, session
                    )
                if should_click_front_layer(top, occluded):
                    self.logger.info("Popup is in front; clicking that popup.")
                    return await click_visible_layer(self, session, x, y)
        except Exception:
            logger.debug("Could not check the front popup before clicking.", exc_info=True)
        return await original_click(self, element_node)

    async def _click_on_coordinate(
        self: Any, coordinate_x: int, coordinate_y: int, force: bool = False
    ) -> dict | None:
        try:
            session = await self.browser_session.get_or_create_cdp_session()
            info = await _point_info(session, coordinate_x, coordinate_y)
            top = str((info or {}).get("tag") or "")
            if top in {"IFRAME", "FRAME"}:
                self.logger.info("Click point is on a popup; clicking that popup.")
                return await click_visible_layer(self, session, coordinate_x, coordinate_y)
        except Exception:
            logger.debug("Could not check the front popup before a coordinate click.", exc_info=True)
        return await original_coordinate(self, coordinate_x, coordinate_y, force)

    DefaultActionWatchdog._click_element_node_impl = _click_element_node_impl  # type: ignore[method-assign]
    DefaultActionWatchdog._click_on_coordinate = _click_on_coordinate  # type: ignore[method-assign]
    _INSTALLED = True


async def click_visible_layer(watchdog: Any, session: Any, x: float, y: float) -> dict[str, float]:
    """Click the painted layer at ``x, y``, stepping into frames that cover the page."""

    browser = watchdog.browser_session
    for _ in range(_MAX_FRAME_DEPTH):
        info = await _point_info(session, x, y)
        if not info or not info.get("frame"):
            break
        child = await _child_session_for_frame(browser, session, x, y, info)
        if child is None or child.session_id == session.session_id:
            break
        x = max(0.0, float(info.get("localX") or 0))
        y = max(0.0, float(info.get("localY") or 0))
        session = child
    aimed = await _control_point(session, x, y)
    if aimed is not None:
        x, y = aimed
        logger.info("Clicking the button inside the popup.")
    return await _dispatch_click(session, x, y)


async def _point_info(session: Any, x: float, y: float) -> dict[str, Any] | None:
    value = await _evaluate(session, _POINT_EXPRESSION, x, y)
    return value if isinstance(value, dict) else None


async def _control_point(session: Any, x: float, y: float) -> tuple[float, float] | None:
    value = await _evaluate(session, _CONTROLS_EXPRESSION, x, y)
    if not isinstance(value, dict):
        return None
    controls = value.get("controls")
    if not isinstance(controls, list):
        return None
    return pick_control(x, y, controls)


async def _evaluate(session: Any, template: str, x: float, y: float) -> Any:
    expression = template % (f"{float(x):.3f}", f"{float(y):.3f}")
    result = await session.cdp_client.send.Runtime.evaluate(
        params={"expression": expression, "returnByValue": True},
        session_id=session.session_id,
    )
    return (result or {}).get("result", {}).get("value")


async def _frame_id_at(session: Any, x: float, y: float) -> str | None:
    client = session.cdp_client
    try:
        await client.send.DOM.enable(session_id=session.session_id)
    except Exception:
        logger.debug("DOM.enable failed before locating the popup frame.", exc_info=True)
    try:
        located = await client.send.DOM.getNodeForLocation(
            params={"x": int(round(x)), "y": int(round(y))},
            session_id=session.session_id,
        )
        backend_node_id = (located or {}).get("backendNodeId")
        if not backend_node_id:
            return None
        described = await client.send.DOM.describeNode(
            params={"backendNodeId": backend_node_id},
            session_id=session.session_id,
        )
        node = (described or {}).get("node") or {}
        name = str(node.get("nodeName") or "").upper()
        if name not in {"IFRAME", "FRAME"}:
            return None
        frame_id = node.get("frameId")
        return str(frame_id) if frame_id else None
    except Exception:
        logger.debug("Could not read the popup frame id.", exc_info=True)
        return None


async def _child_session_for_frame(
    browser: Any, session: Any, x: float, y: float, info: dict[str, Any]
) -> Any | None:
    frame_id = await _frame_id_at(session, x, y)
    if frame_id:
        try:
            return await browser.cdp_client_for_frame(frame_id)
        except Exception:
            logger.debug("Popup frame %s has no session.", frame_id, exc_info=True)

    src = str(info.get("src") or "")
    if not src or src.startswith("about:") or src.startswith("javascript:"):
        return None
    try:
        frames, _sessions = await browser.get_all_frames()
    except Exception:
        logger.debug("Could not list frames for the popup.", exc_info=True)
        return None

    match_id = _matching_frame_id(frames, src)
    if match_id is None:
        return None
    try:
        return await browser.cdp_client_for_frame(match_id)
    except Exception:
        logger.debug("Matched popup frame %s has no session.", match_id, exc_info=True)
        return None


def _positive(info: dict[str, Any] | None, key: str) -> float | None:
    if not info:
        return None
    try:
        value = float(info.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _inside_frame(browser: Any, session: Any) -> bool:
    target_id = getattr(session, "target_id", None)
    manager = getattr(browser, "session_manager", None)
    if not target_id or manager is None or not hasattr(manager, "get_target"):
        return False
    target = manager.get_target(target_id)
    return getattr(target, "target_type", "") in {"iframe", "frame"}


def _urls_match(url: str, src: str) -> bool:
    if url == src:
        return True
    shorter, longer = (url, src) if len(url) <= len(src) else (src, url)
    return len(shorter) >= 12 and longer.startswith(shorter)


def _matching_frame_id(frames: dict[str, dict[str, Any]], src: str) -> str | None:
    """Pick the frame whose URL is the painted iframe, preferring the longest match."""

    best_id: str | None = None
    best_length = -1
    for frame_id, frame in frames.items():
        url = str((frame or {}).get("url") or "")
        if url and _urls_match(url, src) and len(url) > best_length:
            best_id = str(frame_id)
            best_length = len(url)
    return best_id


async def _dispatch_click(session: Any, x: float, y: float) -> dict[str, float]:
    client = session.cdp_client
    session_id = session.session_id
    point = {"x": float(x), "y": float(y)}
    await client.send.Input.dispatchMouseEvent(
        params={"type": "mouseMoved", "pointerType": "mouse", **point},
        session_id=session_id,
    )
    await asyncio.sleep(0.05)
    await client.send.Input.dispatchMouseEvent(
        params={
            "type": "mousePressed",
            "button": "left",
            "buttons": 1,
            "clickCount": 1,
            "pointerType": "mouse",
            **point,
        },
        session_id=session_id,
    )
    await asyncio.sleep(0.05)
    await client.send.Input.dispatchMouseEvent(
        params={
            "type": "mouseReleased",
            "button": "left",
            "buttons": 0,
            "clickCount": 1,
            "pointerType": "mouse",
            **point,
        },
        session_id=session_id,
    )
    return {"click_x": point["x"], "click_y": point["y"]}
