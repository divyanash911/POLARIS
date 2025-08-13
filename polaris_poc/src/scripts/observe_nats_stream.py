#!/usr/bin/env python3
"""
NATS Event Observer
===================

This script connects to a NATS server, subscribes to ALL subjects (">"),
and prints every event in a clean, formatted, and color-coded way for easy monitoring.

Features:
- Wildcard subscription to capture ALL published events
- Pretty-prints JSON payloads with indentation
- Falls back to raw text display if not JSON
- Shows subject, reply, timestamp, and message size
- Color-coded output for readability
"""

import asyncio
import json
from datetime import datetime
import sys

from nats.aio.client import Client as NATS


# Simple ANSI color codes for pretty output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREY = "\033[90m"

async def main():
    nc = NATS()

    server_url = "nats://localhost:4222"  # Change if your server is remote

    await nc.connect(server_url)
    print(f"{Colors.OKGREEN}Connected to NATS at {server_url}{Colors.RESET}")
    print(f"{Colors.OKCYAN}Subscribing to all subjects ('>')...{Colors.RESET}")

    async def message_handler(msg):
        subject = msg.subject
        reply = msg.reply
        data = msg.data.decode(errors="replace")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Try to parse as JSON for pretty output
        try:
            parsed = json.loads(data)
            pretty_data = json.dumps(parsed, indent=2)
            is_json = True
        except json.JSONDecodeError:
            pretty_data = data
            is_json = False

        # Print header
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}--- Event Received ---{Colors.RESET}")
        print(f"{Colors.GREY}Time:    {timestamp}{Colors.RESET}")
        print(f"{Colors.OKGREEN}Subject: {subject}{Colors.RESET}")
        if reply:
            print(f"{Colors.OKCYAN}Reply:   {reply}{Colors.RESET}")
        print(f"{Colors.GREY}Size:    {len(msg.data)} bytes{Colors.RESET}")

        # Print body
        print(f"{Colors.BOLD}Data:{Colors.RESET}")
        if is_json:
            print(f"{Colors.OKCYAN}{pretty_data}{Colors.RESET}")
        else:
            print(f"{Colors.WARNING}{pretty_data}{Colors.RESET}")

        print(f"{Colors.BOLD}{Colors.OKBLUE}{'-'*50}{Colors.RESET}")

    # Wildcard subscription to capture everything
    await nc.subscribe(">", cb=message_handler)

    # Keep the script alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print(f"{Colors.FAIL}\nShutting down...{Colors.RESET}")
        await nc.close()

if __name__ == "__main__":
    asyncio.run(main())
