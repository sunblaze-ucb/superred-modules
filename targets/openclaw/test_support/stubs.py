"""OpenAI-compatible stub upstream servers for live and Docker tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import web


@asynccontextmanager
async def loopback_stub_llm_server(
    *,
    reply: str = "Hello from stub LLM!",
) -> AsyncIterator[str]:
    """Minimal chat-completions stub on loopback (local managed gateway)."""

    async def completions(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": reply}},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def loopback_stub_tool_calling_llm_server(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    final_reply: str = "Tool loop complete.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Stub LLM that issues one tool call on turn 1, then finishes on turn 2."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_arguments),
                        },
                    },
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": final_reply}
            finish_reason = "stop"
        return web.json_response(
            {
                "id": f"chatcmpl-stub-{len(requests)}",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


@asynccontextmanager
async def loopback_recording_stub_llm_server(
    *,
    reply: str = "Stub upstream reply.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Chat-completions stub that records every request body."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        return web.json_response(
            {
                "id": "chatcmpl-record",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": reply}},
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


@asynccontextmanager
async def local_web_page_server(
    *,
    body: str = "ORIGINAL-WEB-CONTENT-NOT-INJECTED",
) -> AsyncIterator[str]:
    """Serve a single static page for ``web_fetch`` live tests."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text=body)

    app = web.Application()
    app.router.add_get("/page", page)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}/page"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def container_stub_llm_server(
    *,
    reply: str = "Docker stub LLM reply.",
) -> AsyncIterator[str]:
    """Stub reachable from inside a container via ``host.docker.internal``."""

    async def completions(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def container_stub_tool_calling_llm_server(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    final_reply: str = "Docker tool loop complete.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Tool-calling stub reachable from inside a container."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_arguments),
                        },
                    },
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": final_reply}
            finish_reason = "stop"
        return web.json_response(
            {
                "id": f"chatcmpl-docker-stub-{len(requests)}",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}", requests
    finally:
        await runner.cleanup()


@asynccontextmanager
async def container_recording_stub_llm_server(
    *,
    reply: str = "Docker stub upstream reply.",
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """Chat-completions stub reachable via ``host.docker.internal`` that
    records every request body (Docker analogue of
    :func:`loopback_recording_stub_llm_server`)."""
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        return web.json_response(
            {
                "id": "chatcmpl-docker-record",
                "object": "chat.completion",
                "choices": [
                    {"message": {"role": "assistant", "content": reply}},
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}", requests
    finally:
        await runner.cleanup()


@asynccontextmanager
async def container_web_page_server(
    *,
    body: str = "ORIGINAL-WEB-CONTENT-NOT-INJECTED",
) -> AsyncIterator[str]:
    """Serve a static page reachable from inside a container via ``host.docker.internal``."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text=body)

    app = web.Application()
    app.router.add_get("/page", page)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 0)  # noqa: S104 - container reachability
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://host.docker.internal:{port}/page"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def loopback_stub_upstream_for_host_proxy(
    *,
    reply: str = "Docker Target stub LLM reply.",
) -> AsyncIterator[str]:
    """Upstream stub for a host-side LLM proxy serving a containerised gateway."""
    async with loopback_stub_llm_server(reply=reply) as url:
        yield url


@asynccontextmanager
async def echo_stub_upstream_for_host_proxy() -> AsyncIterator[str]:
    """Upstream stub that echoes the latest user message in its reply."""

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        user_messages = [
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") == "user"
        ]
        last_user = str(user_messages[-1]) if user_messages else ""
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"ACK:{last_user}",
                        },
                    },
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()
