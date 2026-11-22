# -*- coding: utf-8 -*-
"""Роутер между Claude и доской: задачи и комментарии автора — Claude, а если его нет — агенту GLM.

    python tools/glm/router.py           — без конца, раз в минуту
    python tools/glm/router.py --once    — один шаг (проверка)

Что ждёт ответа: задача в Todo с 🔥 (задача автора) или любая задача доски, где последний комментарий — не «🤖 …»
(комментарий автора), и после него нет ответа «🤖 Claude: / 🤖 GLM:».
Кто отвечает: Claude на месте, если файл присутствия (stand/claude.heartbeat) моложе PRESENT_MINUTES — тогда ждём его
ответа CLAIM_MINUTES, потом отдаём GLM. Claude нет — сразу GLM: главный агент в одном потоке модели, помощник — во втором
(tools/glm/agent.py). GLM пишет ответ комментарием «🤖 GLM: …» в той же задаче; правки кода и прогоны оставляет Claude.
Автор 2026-11-21: «сделай простой роутер между тобой и канбаном, который будет направлять на GLM, если тебя нет».
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "tuning"))
import agent as G  # noqa: E402
import board as B  # noqa: E402
import board_runner as R  # noqa: E402

HEARTBEAT = os.path.join(R.STAND, "claude.heartbeat")
LOG = os.path.join(R.STAND, "router.log")
HANDLED = os.path.join(R.STAND, "router_handled.json")
PRESENT_MINUTES, CLAIM_MINUTES = 1, 15  # автор 2026-11-21: «пусть ждёт не 5 мин, а 1»; метка — каждые 20 с
NOTIFIED: set = set()
ACKED = os.path.join(R.STAND, "router_acked.json")
ACK = "🤖 Роутер: принял"  # быстрое «принял» (задача автора #89; #105 — говорящий первым) — не ответ
OLD_ACK = "🤖 Принял:"  # так «принял» писался до #105


def say(text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as out:
        out.write(line + "\n")


TRIAL = os.path.join(R.STAND, "glm_trial")


def trial() -> bool:
    """Пробный режим (автор 2026-11-21: «проверяй его на реальных задачах»): GLM берёт задачи Очереди и при Claude, в
    полном режиме; Claude проверяет каждую его работу и пишет оценку в задаче."""
    return os.path.exists(TRIAL)


def claude_here() -> bool:
    return os.path.exists(HEARTBEAT) and time.time() - os.path.getmtime(HEARTBEAT) < PRESENT_MINUTES * 60


def ours(text: str) -> bool:
    return (text or "").lstrip().startswith("🤖")


def is_ack(text: str) -> bool:
    return (text or "").lstrip().startswith((ACK, OLD_ACK))


def acknowledge(job: dict) -> None:
    """Сразу, как роутер увидел действие автора (задача автора #89: «если смог отреагировать на действие пользователя —
    сразу пиши стандартный коммент, типа „принял“, возможно более информативный»): кто ответит и когда. Один раз на ключ."""
    acked = set(json.load(open(ACKED, encoding="utf-8"))) if os.path.exists(ACKED) else set()
    if job["key"] in acked or job.get("queue"):
        return
    if claude_here() and not trial():
        who = f"отвечает Claude (он на месте); если не ответит за {CLAIM_MINUTES} мин — GLM"
    elif G.busy() >= G.SLOTS:
        who = "Claude нет, оба потока GLM заняты — GLM ответит, как освободится"
    else:
        who = "Claude нет — отвечает GLM, обычно 1–5 мин"
    what = "комментарий" if job["what"] == "комментарий автора" else "задачу"
    try:
        G.issue_comment(job["number"], f"{ACK}: {what} увидел — {who}." + R.stdout_block(
            R.log_tail(LOG, 6) + [f"{time.strftime('%H:%M:%S')} принял: #{job['number']} {job['what']} → {who}"]), ack=True)
    except Exception as error:  # noqa: BLE001
        say(f"«принял» не оставлен в #{job['number']}: {error!r}")
        return
    acked.add(job["key"])
    json.dump(sorted(acked), open(ACKED, "w", encoding="utf-8"))


def age_minutes(stamp: str) -> float:
    when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60


def pending() -> list[dict]:
    """Задачи, ждущие ответа: (номер, заголовок, что ждёт, с какого времени)."""
    items = B.items("""id createdAt fieldValues(first: 20) { nodes { ... on ProjectV2ItemFieldSingleSelectValue { name
        field { ... on ProjectV2SingleSelectField { name } } } } }
        content { ... on Issue { number title body createdAt comments(last: 20) { nodes { id body createdAt } } } }""")
    handled = set(json.load(open(HANDLED, encoding="utf-8"))) if os.path.exists(HANDLED) else set()
    found = []
    for item in items:
        issue = item.get("content") or {}
        if not issue.get("number"):
            continue
        column = next((v.get("name") for v in item["fieldValues"]["nodes"] if v and v.get("field", {}).get("name") == "Status"), "")
        comments = [c for c in issue["comments"]["nodes"] if not is_ack(c["body"])]  # «принял» — не ответ
        last = comments[-1] if comments else None
        label = next((v.get("name") for v in item["fieldValues"]["nodes"] if v and v.get("field", {}).get("name") == "Метка"), None)
        if column == R.TODO or label:
            continue  # Todo — кухня автора; с меткой («ждёт слова автора», «пауза») — не брать
        if last and not ours(last["body"]):  # комментарий автора — в любой колонке, кроме Todo
            key = f"comment:{last['id']}"
            if key not in handled:
                found.append({"number": issue["number"], "title": issue["title"], "what": "комментарий автора",
                              "text": last["body"], "since": last["createdAt"], "key": key, "order": len(found)})
        elif column == R.QUEUE and "```карточка" not in (issue.get("body") or "") and "```прогон" not in (issue.get("body") or "")                 and not any(ours(c["body"]) for c in comments):
            # Задача автора: в Todo, не карточка прогонщика (нет блока «карточка»), ответа «🤖» ещё нет (🔥 не нужен —
            # 2026-11-21 #62 ждала, пока повисший прогонщик не поставит 🔥).
            key = f"task:{issue['number']}"
            if key not in handled:
                found.append({"number": issue["number"], "title": issue["title"], "what": "задача автора", "text": "",
                              "since": issue["createdAt"], "key": key, "order": len(found)})
        elif column == R.QUEUE and "```карточка" in (issue.get("body") or "") and "```прогон" not in (issue.get("body") or ""):
            # Идея в Очереди без блока «прогон» — правка кода (автор: «если в очереди есть задачи — разбирай их по
            # умолчанию»); GLM берёт её, только когда Claude нет.
            key = f"queue:{issue['number']}"
            if key not in handled:
                found.append({"number": issue["number"], "title": issue["title"], "what": "карточка Очереди с правкой кода",
                              "text": "", "since": issue["createdAt"], "key": key, "queue": True,
                              "priority": R.item_priority({"content": issue}), "order": len(found)})
    # Сначала ответы автору (задачи, комментарии), потом Очередь по приоритету.
    # Сначала ответы на комментарии, потом Очередь — сверху вниз, как на доске (автор: «из очереди бери всегда сверху»).
    return sorted(found, key=lambda f: (f["what"] != "комментарий автора", f["order"]))


def answer(job: dict) -> None:
    """Главный агент GLM (1 поток, помощник — во втором) отвечает в задаче; Claude нет — в полном режиме (правит код,
    проверяет, ставит прогоны). Карточка Очереди с правкой кода — делает её целиком."""
    if job.get("queue"):
        task = (f"Claude нет, ты его заменяешь. В Очереди доски карточка #{job['number']} «{job['title']}» — правка кода. "
                "Прочитай карточку (issue_read), найди ответ в статье и коде, сделай правку новой настройкой (по умолчанию "
                "выключена), проверь на CPU, что по умолчанию всё бит в бит прежнее, и py_compile, поставь карточке блок "
                "«прогон» (card_run) с включённой настройкой. Ответь комментарием в задаче: что сделал, где, как проверил. "
                "Если правка слишком большая или рискованная — сделай безопасную часть, остальное опиши и оставь Claude.")
        G.run(task, full=True, helper=True, max_steps=80, slot_wait=0)
        return
    task = (f"На доске «Подбор параметров DAMP» тебя ждёт {job['what']} в задаче #{job['number']} «{job['title']}».\n"
            + (f"Комментарий автора: {job['text']}\n" if job["text"] else "")
            + "Прочитай задачу (issue_read), при нужде — доску, файлы репозитория (docs/DIRECTION.md, docs/TUNING.md, "
              "tools/tuning/board_runner.py) или помощника (subagent). Ответь автору комментарием в ЭТОЙ задаче "
              "(issue_comment): что понял, что сделал или что предлагаешь. Если задача простая и решается доской "
              "(приоритет, перенос между Todo / Очередью / Хорошими) — сделай. Если нужна правка кода или прогоны — "
              "напиши, что это для Claude, задачу оставь в Todo.")
    before = len(G.issue(job["number"])["comments"]["nodes"])
    result = G.run(task, full=not claude_here() or trial(), board_tools=True, helper=True, max_steps=60, slot_wait=0)
    after = G.issue(job["number"])["comments"]["nodes"]
    if len(after) == before or not any(ours(c["body"]) for c in after[before:]):
        G.issue_comment(job["number"], result or "не смог ответить — оставляю Claude.")


def step() -> str:
    jobs = pending()
    if not jobs:
        return "ждать нечего"
    here = claude_here() and not trial()
    handled = set(json.load(open(HANDLED, encoding="utf-8"))) if os.path.exists(HANDLED) else set()
    for job in jobs:
        acknowledge(job)
        if here and (job.get("queue") or age_minutes(job["since"]) < CLAIM_MINUTES):
            if job["key"] not in NOTIFIED and not job.get("queue"):
                NOTIFIED.add(job["key"])
                say(f"ждёт Claude: #{job['number']} {job['what']} «{job['title']}»")  # сигнал Claude (Monitor)
            continue  # Claude на месте — его очередь отвечать; Очередь при Claude — его
        if G.busy() >= G.SLOTS:
            return "оба потока GLM заняты"
        why = "проба при Claude" if trial() else ("Claude не ответил за " + str(CLAIM_MINUTES) + " мин" if here else "Claude нет")
        say(f"GLM берёт: #{job['number']} {job['what']} ({why})")
        try:
            answer(job)
            handled.add(job["key"])
            json.dump(sorted(handled), open(HANDLED, "w", encoding="utf-8"))
            say(f"GLM ответил: #{job['number']}")
        except Exception as error:  # noqa: BLE001
            say(f"GLM не справился с #{job['number']}: {error!r}")
        return "ответ"
    return f"ждёт Claude: {len(jobs)}"


def main() -> None:
    say("роутер запущен")
    last = None
    while True:
        try:
            status = step()
        except Exception as error:  # noqa: BLE001
            status = f"ошибка шага: {error!r}"
        if status != last:
            say(status)
            last = status
        if "--once" in sys.argv:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
