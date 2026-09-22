"""
Discord webhook notifications.

The webhook URL is a bearer secret: anyone holding it can post to your channel.
It therefore lives in the macOS Keychain and never in this file, never in the
repo, never in a log line — the same handling the ntfy topic had.

    store:  security add-generic-password -a "$USER" -s discord-alpha-webhook \
              -w 'https://discord.com/api/webhooks/...' -U
    read :  security find-generic-password -a "$USER" -s discord-alpha-webhook -w

If no webhook is configured, alerts fall back to a local macOS notification so
a missing secret degrades the channel rather than losing the alert.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import urllib.error
import urllib.request

from config import (
    DISCORD_KEYCHAIN_SERVICE, DISCORD_USERNAME, DISCORD_COLOR_EXECUTE,
    DISCORD_COLOR_APPROACH, DISCORD_COLOR_EXTENDED, DISCORD_COLOR_INFO,
)
from execution import State

_MAX_FIELD = 1024      # Discord's per-field character cap
_MAX_EMBEDS = 10       # Discord's per-message embed cap


def webhook_url() -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""),
             "-s", DISCORD_KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=10)
        url = out.stdout.strip()
        return url if url.startswith("https://") else None
    except Exception:
        return None


def _local_alert(title: str, message: str) -> None:
    try:
        safe_t = title.replace('"', "'")
        safe_m = message.replace('"', "'")[:200]
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe_m}" with title "{safe_t}" sound name "Ping"'],
            capture_output=True, timeout=10)
    except Exception:
        pass


def _post(payload: dict) -> bool:
    url = webhook_url()
    if not url:
        return False
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "alpha-engine/1.0"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        # Never let the URL itself reach a log or a traceback.
        print(f"discord: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"discord: {type(e).__name__}")
        return False


def _clip(s: str, n: int = _MAX_FIELD) -> str:
    s = s or "—"
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- embeds
def _colour(state: State) -> int:
    return {State.EXECUTE: DISCORD_COLOR_EXECUTE,
            State.APPROACHING: DISCORD_COLOR_APPROACH,
            State.EXTENDED: DISCORD_COLOR_EXTENDED}.get(state, DISCORD_COLOR_INFO)


def _one_liner(summary: str, limit: int = 240) -> str:
    """First sentence or two of the business description."""
    s = " ".join((summary or "").split())
    if not s:
        return "—"
    out, total = [], 0
    for sent in s.split(". "):
        if total + len(sent) > limit and out:
            break
        out.append(sent)
        total += len(sent) + 2
    return _clip(". ".join(out).rstrip(".") + ".", limit + 20)


def build_embed(snap: dict, plan, extra_note: str = "") -> dict:
    """
    One alert card. Every section the brief asks for, in decision order:
    what the company is, why it is fundamentally acceptable, where price is
    versus the range, what to do, how much, and where to get out.
    """
    sym = snap.get("symbol", "?")
    name = snap.get("name", sym)
    state = plan.state

    verdict = {
        State.EXECUTE: "🟢 **EXECUTE NOW** — price is inside the ideal range.",
        State.APPROACHING: "🟡 **WAIT / ARM** — below trigger, not yet a buy.",
        State.EXTENDED: "🟠 **DO NOT CHASE** — extended past the ideal range.",
        State.BELOW: "⚪️ **NO ACTION** — price is away from the setup.",
    }[state]

    # ---- fundamentals
    def _pe(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) and v else "n/a"

    fcf = snap.get("fcf")
    fcf_txt = "n/a"
    if isinstance(fcf, (int, float)) and fcf:
        fcf_txt = f"${fcf/1e9:.2f}B"
        fy = snap.get("fcf_yield")
        if fy:
            fcf_txt += f" ({fy*100:.1f}% yield)"

    fund = (f"**P/E** {_pe(snap.get('trailing_pe'))} · "
            f"**Fwd P/E** {_pe(snap.get('forward_pe'))}\n"
            f"**FCF** {fcf_txt}\n")
    de, cr, roe = snap.get("debt_equity"), snap.get("current_ratio"), snap.get("roe")
    health = []
    if de is not None:
        health.append(f"D/E {de:.2f}x")
    if cr is not None:
        health.append(f"current {cr:.2f}")
    if roe is not None:
        health.append(f"ROE {roe*100:.0f}%")
    fund += "**Health** " + (" · ".join(health) if health else "limited data")
    if snap.get("dcf_upside") is not None:
        # Labelled "rel." because this single-stage model runs systematically
        # low (median about -60% across the watchlist): a 9% WACC with a capped
        # growth rate cannot justify a quality compounder's multiple. It ranks
        # names against each other well; its absolute level is not a fair value.
        fund += (f"\n**DCF (rel.)** ${snap['dcf_value']:.0f} "
                 f"({snap['dcf_upside']*100:+.0f}% vs spot)")
    if snap.get("comp_discount") is not None:
        fund += f"\n**Comps** {snap['comp_discount']*100:+.0f}% vs peer median P/E"

    # ---- price vs range
    px = (f"**Live** ${plan.price:.2f}\n"
          f"**Ideal range** ${plan.ideal_low:.2f} – ${plan.ideal_high:.2f}\n"
          f"**Trigger** ${plan.trigger:.2f}  ({plan.pct_from_trigger*100:+.2f}% away)")

    fields = [
        {"name": "What it does", "value": _clip(_one_liner(snap.get("summary", ""))),
         "inline": False},
        {"name": "Fundamental snapshot", "value": _clip(fund), "inline": True},
        {"name": "Price vs ideal range", "value": _clip(px), "inline": True},
        {"name": "Decision", "value": _clip(verdict), "inline": False},
    ]

    if plan.shares > 0:
        alloc = (f"**{plan.shares:g} shares** ≈ **${plan.dollars:,.2f}**\n"
                 f"Risk ${plan.risk_dollars:,.2f} → reward ${plan.reward_dollars:,.2f} "
                 f"({plan.rr:.1f}R)\n*{plan.size_note}*")
        exits = (f"**Target** ${plan.target:.2f}  (+{(plan.target/plan.price-1)*100:.2f}%)\n"
                 f"**Stop** ${plan.stop:.2f}  (−{(1-plan.stop/plan.price)*100:.2f}%)\n"
                 f"**Flat by** 15:55 ET — nothing held overnight")
        # A DCF BELOW spot is not a target — quoting it as one would tell you to
        # sell a $57 stock at $27. Above spot it is context for the upside;
        # below spot it is a caution, and it is labelled as one.
        if plan.valuation_target:
            vt = plan.valuation_target
            if vt > plan.price:
                exits += (f"\n*DCF ${vt:.0f} ({vt/plan.price-1:+.0%}) — "
                          f"valuation supports the move*")
            else:
                exits += (f"\n*DCF ${vt:.0f} ({vt/plan.price-1:+.0%}) — momentum "
                          f"trade, not a value entry*")
        fields.append({"name": "Capital allocation", "value": _clip(alloc), "inline": True})
        fields.append({"name": "Targets & exits", "value": _clip(exits), "inline": True})

    if extra_note:
        fields.append({"name": "Notes", "value": _clip(extra_note), "inline": False})

    sector = snap.get("sector", "—")
    industry = snap.get("industry", "—")

    return {
        "title": f"{sym} · {name}",
        "description": f"*{sector} — {industry}*",
        "color": _colour(state),
        "fields": fields,
        "footer": {"text": f"Alpha Engine · {state.value}"},
    }


# ---------------------------------------------------------------- senders
def send_alerts(embeds: list[dict], header: str = "") -> bool:
    """Post up to 10 embeds per message, chunked automatically."""
    if not embeds:
        return True
    ok = True
    for i in range(0, len(embeds), _MAX_EMBEDS):
        chunk = embeds[i:i + _MAX_EMBEDS]
        payload = {"username": DISCORD_USERNAME, "embeds": chunk}
        if header and i == 0:
            payload["content"] = header[:2000]
        if not _post(payload):
            ok = False
            titles = ", ".join(e.get("title", "?").split(" · ")[0] for e in chunk)
            _local_alert("Alpha Engine (Discord failed)", titles)
    return ok


def send_text(title: str, message: str, colour: int = DISCORD_COLOR_INFO) -> bool:
    payload = {
        "username": DISCORD_USERNAME,
        "embeds": [{"title": title, "description": _clip(message, 4000), "color": colour}],
    }
    if _post(payload):
        return True
    _local_alert(title, message)
    return False


def configured() -> bool:
    return webhook_url() is not None


SETUP_HELP = textwrap.dedent("""\
    Discord webhook is not configured.

    1. In Discord: Server Settings → Integrations → Webhooks → New Webhook.
       Pick the channel, then "Copy Webhook URL".
    2. Store it in the Keychain (paste it into YOUR terminal, not into chat):

         security add-generic-password -a "$USER" -s discord-alpha-webhook \\
           -w 'PASTE_URL_HERE' -U

    3. Verify:  python3 notify.py --test
""")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        if not configured():
            print(SETUP_HELP)
            sys.exit(1)
        ok = send_text("Alpha Engine — connection test",
                       "Webhook is live. Alerts will arrive in this channel.",
                       DISCORD_COLOR_EXECUTE)
        print("sent" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    print("configured" if configured() else SETUP_HELP)
