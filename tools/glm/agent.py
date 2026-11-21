# -*- coding: utf-8 -*-
"""Агент на модели GLM 5.3 (сервер автора, vLLM, 2 потока) — для простых задач.

    python tools/glm/agent.py status                       — туннель, модель, занятые потоки
    python tools/glm/agent.py ask "задача" [--board]       — выполнить задачу (только чтение репозитория; --board — ещё и
                                                             доска: прочитать, комментарий в задаче, приоритет, колонка)
    python tools/glm/agent.py ask "задача" --full          — полный режим (замена Claude): ещё запись файлов, команды
                                                             bash с запретами, блок «прогон» и новые карточки, помощник

Модель слушает 127.0.0.1:8893 на сервере (GML_SSH_HOST в .env, пароль — PASS_SSH_PASS): агент сам поднимает SSH-туннель
на 127.0.0.1:18893. У сервера 2 потока (vLLM --max-num-seqs 2) — они поделены на 2 слота (файлы-замки): Claude может
держать до 2 агентов GLM сразу; главный агент роутера (когда Claude нет) занимает 1 слот и может позвать 1 помощника во
второй (инструмент subagent). Автор 2026-11-21: «сделай простой роутер между тобой и канбаном, который будет направлять
на GLM, если тебя нет, и даст возможность тебе (2 потока / саб-агента) или GLM (1 поток / саб-агент) использовать как
агента для простых задач».

Безопасность: агент не пишет файлы и не выполняет команды; на доске — только комментарий «🤖 GLM: …», эмодзи
приоритета в заголовке и перенос между Todo / Очередью / Хорошими (Прогон и Анализ ведёт прогонщик).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "tuning"))

LOCAL_PORT, REMOTE_PORT = 18893, 8893
API = f"http://127.0.0.1:{LOCAL_PORT}/v1"
SLOTS_DIR = os.path.join(tempfile.gettempdir(), "damp-glm-slots")
SLOTS = 2
MARK = "🤖 GLM:"
# Python проекта: DAMP_PYTHON из окружения или .env, иначе тот, которым запущен агент (docs/HARNESS.md).
PYTHON = (os.environ.get("DAMP_PYTHON") or sys.executable).replace("\\", "/")  # уточняется из .env ниже


def env() -> dict:
    values = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


if not os.environ.get("DAMP_PYTHON") and os.path.exists(os.path.join(ROOT, ".env")) and env().get("DAMP_PYTHON"):
    PYTHON = env()["DAMP_PYTHON"].replace("\\", "/")


# ---------- туннель ----------

def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", LOCAL_PORT)) == 0


def ssh_binary() -> str:
    for path in (r"C:\Program Files\Git\usr\bin\ssh.exe", shutil.which("ssh") or ""):
        if path and os.path.exists(path):
            return path
    raise RuntimeError("нет ssh")


def ensure_tunnel() -> None:
    """SSH-туннель 127.0.0.1:18893 → сервер 127.0.0.1:8893 (пароль — через askpass.sh из .env, на экран не выводится)."""
    if port_open():
        return
    host = env().get("GML_SSH_HOST")
    if not host:
        raise RuntimeError("в .env нет GML_SSH_HOST")
    askpass = os.path.join(HERE, "askpass.sh")
    environ = {**os.environ, "SSH_ASKPASS": askpass.replace("\\", "/"), "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0",
               "DAMP_ENV": os.path.join(ROOT, ".env").replace("\\", "/")}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([ssh_binary(), "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                      "-o", "StrictHostKeyChecking=accept-new", "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                      "-L", f"{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}", host], env=environ, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    for _ in range(30):
        if port_open():
            return
        time.sleep(0.5)
    raise RuntimeError("туннель к модели не поднялся")


# ---------- потоки (слоты) ----------

class Slot:
    """Один из 2 потоков модели: файл-замок с pid; брошенный (процесс умер) — освобождается."""

    def __init__(self, wait: float = 0.0) -> None:
        self.path = None
        os.makedirs(SLOTS_DIR, exist_ok=True)
        deadline = time.time() + wait
        while self.path is None:
            for index in range(SLOTS):
                path = os.path.join(SLOTS_DIR, f"slot{index}.lock")
                if os.path.exists(path) and not alive(open(path).read().strip()):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    self.path = path
                    break
                except FileExistsError:
                    continue
            if self.path is None:
                if time.time() >= deadline:
                    raise RuntimeError("оба потока GLM заняты")
                time.sleep(2)

    def __enter__(self) -> "Slot":
        return self

    def __exit__(self, *exc) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


def alive(pid: str) -> bool:
    try:
        pid_number = int(pid)
    except ValueError:
        return False
    if os.name == "nt":
        found = subprocess.run(["tasklist", "/FI", f"PID eq {pid_number}", "/NH"], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout or b""
        return str(pid_number).encode() in found  # вывод tasklist — в кодировке консоли (cp866), сравниваем байты
    try:
        os.kill(pid_number, 0)
        return True
    except OSError:
        return False


def busy() -> int:
    if not os.path.isdir(SLOTS_DIR):
        return 0
    return sum(1 for name in os.listdir(SLOTS_DIR) if name.endswith(".lock")
               and alive(open(os.path.join(SLOTS_DIR, name)).read().strip()))


# ---------- модель ----------

def model_name() -> str:
    with urllib.request.urlopen(f"{API}/models", timeout=10) as response:
        return json.load(response)["data"][0]["id"]


def chat(messages: list[dict], tools: list[dict] | None, effort: str = "medium") -> dict:
    body = {"model": model_name(), "messages": messages, "max_tokens": 16384, "temperature": 0.3,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(f"{API}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)["choices"][0]["message"]


# ---------- инструменты ----------

def scratch_path(path: str) -> str | None:
    """Путь во временной папке GLM: «scratch/…» или абсолютный внутри неё; иначе None."""
    base = os.path.realpath(os.path.join(tempfile.gettempdir(), "damp-glm-scratch"))
    if path.replace("\\", "/").startswith("scratch/"):
        return os.path.join(base, path[8:])
    full = os.path.realpath(path) if os.path.isabs(path) else None
    return full if full and full.startswith(base) else None


def inside(path: str) -> str:
    found = scratch_path(path)
    if found:
        return found
    full = os.path.realpath(os.path.join(ROOT, path))
    if not full.startswith(os.path.realpath(ROOT)) or os.path.basename(full) == ".env":
        raise ValueError("только файлы репозитория (и не .env)")
    return full


def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    lines = open(inside(path), encoding="utf-8", errors="replace").read().split("\n")
    start = max(1, int(offset))
    part = lines[start - 1:start - 1 + min(int(limit), 400)]
    return "\n".join(f"{start + i}: {line[:400]}" for i, line in enumerate(part)) + f"\n[строк всего: {len(lines)}]"


def list_dir(path: str = ".") -> str:
    full = inside(path)
    return "\n".join(sorted(name + ("/" if os.path.isdir(os.path.join(full, name)) else "") for name in os.listdir(full)
                            if not name.startswith(".git") and name != ".env"))[:6000]


def grep(pattern: str, path: str = ".", glob: str = "") -> str:
    found, rx = [], re.compile(pattern)
    for folder, dirs, files in os.walk(inside(path)):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "data", "node_modules")]
        for name in files:
            if name == ".env" or (glob and not re.fullmatch(glob.replace(".", r"\.").replace("*", ".*"), name)):
                continue
            file = os.path.join(folder, name)
            try:
                for number, line in enumerate(open(file, encoding="utf-8", errors="ignore"), 1):
                    if rx.search(line):
                        found.append(f"{os.path.relpath(file, ROOT)}:{number}: {line.strip()[:200]}")
                        if len(found) >= 80:
                            return "\n".join(found) + "\n[обрезано на 80]"
            except OSError:
                continue
    return "\n".join(found) or "ничего"


def board_read() -> str:
    import board as B
    project = B.board()
    status = B.status_field(project)["name"]
    lines = []
    for item in project["items"]["nodes"]:
        content = item.get("content") or {}
        column = next((v.get("name") for v in item["fieldValues"]["nodes"] if v and v.get("field", {}).get("name") == status), "")
        lines.append(f"[{column}] #{content.get('number', '—')} {content.get('title', '')}")
    return "\n".join(lines)


def issue(number: int) -> dict:
    import board as B
    found = B.graphql("""query($o: String!, $n: String!, $i: Int!) { repository(owner: $o, name: $n) { issue(number: $i) {
        id number title body state comments(last: 30) { nodes { body createdAt } } } } }""",
                      {"o": B.OWNER, "n": "DAMP", "i": int(number)})["repository"]["issue"]
    if not found:
        raise ValueError(f"нет задачи #{number}")
    return found


def issue_read(number: int) -> str:
    found = issue(number)
    body = re.sub(r"\n### Дерево прогонов.*", "", found.get("body") or "", flags=re.S)
    body = re.sub(r"<details>.*?</details>", "", body, flags=re.S)
    comments = "\n\n".join(f"— {c['createdAt']}: {c['body']}" for c in found["comments"]["nodes"])
    return f"#{found['number']} {found['title']} [{found['state']}]\n\n{body[:6000]}\n\nКомментарии:\n{comments[:6000]}"


def issue_comment(number: int, text: str) -> str:
    import board as B
    found = issue(number)
    last = (found["comments"]["nodes"] or [{}])[-1].get("body") or ""
    if last.lstrip().startswith("🤖"):  # пока GLM думал, ответил Claude (или прогонщик) — второй ответ не нужен
        return "комментарий не оставлен: на последний комментарий автора уже ответили"
    text = re.sub(r"^\s*(🤖\s*(Claude|GLM|Прогонщик)\s*:\s*)+", "", text)  # модель копирует чужую подпись
    body = f"{MARK} {text}"
    B.graphql("""mutation($i: ID!, $b: String!) { addComment(input: {subjectId: $i, body: $b}) { clientMutationId } }""",
              {"i": found["id"], "b": body})
    return "комментарий оставлен"


def card(number: int) -> tuple[dict, dict]:
    import board as B
    project = B.board()
    for item in project["items"]["nodes"]:
        if (item.get("content") or {}).get("number") == int(number):
            return project, item
    raise ValueError(f"на доске нет карточки #{number}")


def card_priority(number: int, emoji: str) -> str:
    import board_runner as R
    if emoji not in R.PRIORITY_EMOJI.values():
        raise ValueError("приоритет — одно из 🔥 ⬆️ ➡️ ⬇️")
    _, item = card(number)
    content = item["content"]
    R.save_card(content, title=f"{emoji} {R.TITLE_MARKS.sub('', content['title'])}")
    return "приоритет поставлен"


def card_move(number: int, column: str) -> str:
    import board_runner as R
    if column not in (R.TODO, R.QUEUE, R.GOOD):
        raise ValueError("переносить можно только в Todo, Очередь или Хорошие")
    project, item = card(number)
    R.move(project, item, column)
    return f"карточка #{number} → {column}"


JOURNAL_DIR = os.path.join(tempfile.gettempdir(), "damp-glm-logs")
JOURNAL = os.path.join(JOURNAL_DIR, time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}.jsonl")


def journal(entry: dict) -> None:
    """Журнал шагов агента (что сказал, какие инструменты, что вернули) — для разбора, что GLM умеет."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as out:
        out.write(json.dumps({"time": time.strftime("%H:%M:%S"), **entry}, ensure_ascii=False) + "\n")


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


S, I = {"type": "string"}, {"type": "integer"}
READ_TOOLS = {
    "read_file": (read_file, schema("read_file", "Прочитать файл репозитория DAMP (строки offset…offset+limit).",
                                    {"path": S, "offset": I, "limit": I}, ["path"])),
    "list_dir": (list_dir, schema("list_dir", "Список файлов папки репозитория.", {"path": S}, [])),
    "grep": (grep, schema("grep", "Поиск по регулярному выражению в файлах репозитория (glob — маска имени, напр. *.py).",
                          {"pattern": S, "path": S, "glob": S}, ["pattern"])),
}
BOARD_TOOLS = {
    "board_read": (board_read, schema("board_read", "Все карточки доски подбора: [колонка] #номер заголовок.", {}, [])),
    "issue_read": (issue_read, schema("issue_read", "Задача (карточка) по номеру: текст и комментарии.", {"number": I}, ["number"])),
    "issue_comment": (issue_comment, schema("issue_comment", "Оставить комментарий в задаче (ответ автору).",
                                            {"number": I, "text": S}, ["number", "text"])),
    "card_priority": (card_priority, schema("card_priority", "Поставить приоритет карточке: 🔥 ⬆️ ➡️ ⬇️.",
                                            {"number": I, "emoji": S}, ["number", "emoji"])),
    "card_move": (card_move, schema("card_move", "Перенести карточку в колонку Todo, Очередь или Хорошие.",
                                    {"number": I, "column": S}, ["number", "column"])),
}

SYSTEM = """Ты — GLM, помощник в проекте DAMP (иерархия карт кодов по статье; подбор параметров на доске GitHub
«Подбор параметров DAMP»: колонки Todo → Очередь → Прогон → Анализ → Хорошие / Плохие; карточки — задачи репозитория
qwertukg/DAMP; приоритет — эмодзи в начале заголовка 🔥 ⬆️ ➡️ ⬇️; Прогон и Анализ ведёт прогонщик, их не трогай).
Отвечай по-русски, коротко и по делу. Работай инструментами, факты бери из файлов и доски, не выдумывай. Ты не можешь
менять файлы и запускать команды: если задача требует правки кода или прогонов, опиши, что нужно сделать, и скажи, что
это для Claude. Закончив, дай итоговый ответ обычным текстом без вызова инструментов."""

# ---------- полный режим: GLM заменяет Claude (автор: «GLM должен уметь полностью тебя заменять, если ты выключен, а
# туннель работает») ----------

SCRATCH = os.path.join(tempfile.gettempdir(), "damp-glm-scratch")
DENY = re.compile(
    r"\b(curl|wget|ssh|scp|sftp|ping|nmap|nc|netcat|telnet|ftp|Invoke-WebRequest|Invoke-RestMethod|iwr|irm|Start-BitsTransfer)\b"
    r"|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    r"|\bgit\s+(push|commit|reset|rebase|checkout|clean|stash|merge|tag|branch|remote|switch|restore|am|cherry-pick)\b"
    r"|\brm\s+-[a-z]*r|\brmdir\b|\bdel\b|\bformat\b|\bshutdown\b|\breboot\b|Stop-Process|taskkill|\bkill\b"
    r"|\bpip\s+install|\bnpm\s+(i|install)\b|\buv\s+pip"
    r"|tune_cycle|tune_pair|board_runner|router\.py|\.env\b", re.I)


def writable(path: str) -> str:
    """Писать можно в репозиторий (не .env, не .git) и во временную папку GLM."""
    found = scratch_path(path)
    if found:
        os.makedirs(os.path.dirname(found), exist_ok=True)
        return found
    full = inside(path)
    if os.sep + ".git" + os.sep in full + os.sep:
        raise ValueError("в .git писать нельзя")
    return full


def write_file(path: str, content: str) -> str:
    full = writable(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w", encoding="utf-8", newline="\n").write(content)
    return f"записано {len(content)} символов в {path}"


def edit_file(path: str, old: str, new: str) -> str:
    full = writable(path)
    text = open(full, encoding="utf-8").read()
    count = text.count(old)
    if count != 1:
        raise ValueError(f"старый текст найден {count} раз — нужен ровно 1 (дай больше окружения)")
    open(full, "w", encoding="utf-8", newline="\n").write(text.replace(old, new))
    return "заменено"


def run_command(command: str, timeout: int = 300) -> str:
    """Команда bash в корне репозитория (Python проекта — DAMP_PYTHON из .env). Сеть, git-запись,
    удаление, убийство процессов, установка пакетов и запуск стенда — запрещены."""
    if DENY.search(command):
        raise ValueError("команда запрещена правилами (сеть, git-запись, удаление, процессы, пакеты, стенд, .env)")
    bash = r"C:\Program Files\Git\bin\bash.exe" if os.path.exists(r"C:\Program Files\Git\bin\bash.exe") else "bash"
    done = subprocess.run([bash, "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=min(int(timeout), 900), env={**os.environ, "RERUN": "off", "PYTHONIOENCODING": "utf-8"},
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = (done.stdout or "") + ("\n[stderr]\n" + done.stderr if done.stderr else "")
    return f"код выхода {done.returncode}\n" + (out[-10000:] if len(out) > 10000 else out)


def card_run(number: int, changes: str, label: str, priority: int = 1) -> str:
    """Поставить карточке блок «прогон» (прогонщик возьмёт её сам)."""
    import board_runner as R
    _, item = card(number)
    spec = {"changes": json.loads(changes), "label": label, "priority": int(priority)}
    body = item["content"].get("body") or ""
    tree = R.TREE_BLOCK.search(body)
    R.set_body(item, R.compose(R.card_meta(item), json.dumps(spec, ensure_ascii=False), None, [], "", tree.group(0) if tree else ""))
    return "блок «прогон» поставлен"


def card_create(what: str, changes: str, label: str, priority: int = 0) -> str:
    """Новая карточка в Очередь: что пробуем (подробно, с § статьи и ожиданием) и блок «прогон»."""
    import board as B
    import board_runner as R
    spec = {"changes": json.loads(changes), "label": label, "priority": int(priority)}
    R.add_card(B.board(), label, f"**Что:** {what}\n\n```прогон\n{json.dumps(spec, ensure_ascii=False)}\n```")
    return "карточка создана"


FULL_TOOLS = {
    "write_file": (write_file, schema("write_file", "Записать файл (репозиторий или scratch/… — временная папка).",
                                      {"path": S, "content": S}, ["path", "content"])),
    "edit_file": (edit_file, schema("edit_file", "Заменить в файле ровно одно вхождение old на new.",
                                    {"path": S, "old": S, "new": S}, ["path", "old", "new"])),
    "run_command": (run_command, schema("run_command", "Команда bash в корне репозитория (проверки на CPU, py_compile, "
                                                       "маленькие скрипты; python проекта — " + PYTHON + " -X utf8).",
                                        {"command": S, "timeout": I}, ["command"])),
    "card_run": (card_run, schema("card_run", "Поставить карточке блок «прогон»: changes — JSON настроек, label — коротко, "
                                              "priority 0–3.", {"number": I, "changes": S, "label": S, "priority": I},
                                  ["number", "changes", "label"])),
    "card_create": (card_create, schema("card_create", "Новая карточка в Очередь с блоком «прогон».",
                                        {"what": S, "changes": S, "label": S, "priority": I}, ["what", "changes", "label"])),
}

SYSTEM_FULL = SYSTEM.replace(
    "Ты не можешь\nменять файлы и запускать команды: если задача требует правки кода или прогонов, опиши, что нужно сделать, и скажи, что\nэто для Claude.",
    "Claude сейчас нет — ты делаешь его работу целиком.") + """

Правила проекта (как у Claude):
- Сначала статья: ищи ответ в docs/DAMP_extracted.txt и docs/DAMP_continuation.md (продолжение — тот же вес), цитату с §
  клади рядом с решением. Направление подбора — docs/DIRECTION.md, ход — docs/TUNING.md, правила доски — README доски.
- Правка кода — новой настройкой в client/config.toml с комментарием и тегом [КАНОН] / [РАСШИРЕНИЕ] / [НЕ КАНОН]; по
  умолчанию выключена, чтобы опора не менялась. Проверь на CPU маленьким скриптом, что по умолчанию результат бит в бит
  прежний, и py_compile. Прогоны сам не запускай — поставь карточке блок «прогон» (card_run), прогонщик возьмёт.
- Правь файлы маленькими кусками: edit_file (старый текст → новый), а не переписывай файл целиком; копию файла делай
  командой cp (run_command). Не описывай правку словами — делай её инструментами, потом проверяй командой.
- Ничего не коммить и не пушить. Не трогай видеокарту, стенд, прогонщик и роутер. Не удаляй файлы.
- Отвечай в задаче комментарием (issue_comment) коротко: что сделал, где (файл:строка), что дальше.
- Не уверен или задача большая и рискованная — сделай безопасную часть, остальное опиши и оставь Claude (задача в Todo).
"""


def run(task: str, *, board_tools: bool = False, helper: bool = False, full: bool = False, max_steps: int = 30, effort: str = "medium",
        slot_wait: float = 0.0) -> str:
    """Агентский цикл на одном потоке модели: задача → вызовы инструментов → итоговый ответ."""
    ensure_tunnel()
    tools = dict(READ_TOOLS)
    if board_tools or full:
        tools.update(BOARD_TOOLS)
    if full:
        tools.update(FULL_TOOLS)
    if helper:
        def subagent(task: str) -> str:
            try:
                return run(task, board_tools=False, helper=False, max_steps=20, effort=effort, slot_wait=0)
            except RuntimeError as error:
                return f"помощник не запущен: {error}"
        tools["subagent"] = (subagent, schema("subagent", "Поручить помощнику (второй поток модели, только чтение "
                                                          "репозитория) подзадачу; вернёт его ответ.", {"task": S}, ["task"]))
    messages = [{"role": "system", "content": SYSTEM_FULL if full else SYSTEM}, {"role": "user", "content": task}]
    with Slot(wait=slot_wait):
        for _ in range(max_steps):
            message = chat(messages, [t[1] for t in tools.values()], effort)
            calls = message.get("tool_calls") or []
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls} if calls
                            else {"role": "assistant", "content": message.get("content") or ""})
            journal({"step": len(messages), "content": (message.get("content") or "")[:2000],
                     "reasoning": (message.get("reasoning_content") or message.get("reasoning") or "")[:2000],
                     "calls": [c["function"] for c in calls]})
            if not calls:
                answer = (message.get("content") or "").strip()
                return answer or (message.get("reasoning_content") or message.get("reasoning") or "").strip()
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"].get("arguments") or "{}")
                    result = str(tools[name][0](**arguments)) if name in tools else f"нет инструмента {name}"
                except Exception as error:  # noqa: BLE001  ошибку — модели, пусть поправится
                    result = f"ошибка: {error}"
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result[:12000]})
                journal({"tool": name, "result": result[:1500]})
        return "не уложился в число шагов; последнее: " + (messages[-1].get("content") or "")[:2000]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "ask"):
        sys.exit(__doc__)
    if sys.argv[1] == "status":
        ensure_tunnel()
        print(f"туннель: {'есть' if port_open() else 'нет'}; модель: {model_name()}; занято потоков: {busy()} из {SLOTS}")
        return
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    print(run(" ".join(args), board_tools="--board" in sys.argv, full="--full" in sys.argv, helper="--full" in sys.argv,
              max_steps=60 if "--full" in sys.argv else 30, slot_wait=600))


if __name__ == "__main__":
    main()
