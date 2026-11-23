# -*- coding: utf-8 -*-
"""Прогонщик доски: сам гоняет карточки плана по флоу доски, пока на них нет метки (ждёт слова автора / пауза).

    python tools/tuning/board_runner.py          — без конца: брать карточки, гонять пары, писать итоги, двигать по колонкам
    python tools/tuning/board_runner.py --once   — один шаг (проверка)

Карточка берётся из «Очередь» сверху вниз, если на ней нет поля «Метка» и в тексте есть блок

    ```прогон
    {"changes": {"НАСТРОЙКА": значение, ...}, "label": "коротко, что пробуем"}
    ```

Флоу (из описаний колонок): «Прогон» — идёт пара; «Анализ прогона» — итог пары записан, решается дальше; «Хорошие варианты» /
«Плохие варианты» — по правилу последовательного теста стратегии v3 (docs/CLUSTERS.md) мерой М1u (среднее М1 по цифрам и по
3 раскладкам верха): этап 1 — зёрна 1/3: среднее Δ < +0.03 или оба ≤ 0 → плохие; этап 2 — зёрна 0/2: по 4 зёрнам Δ ≥ +0.06 и
не меньше 3 в плюсе → хорошие, +0.04…0.06 → этап 3 (зёрна 4/5): по 6 зёрнам Δ ≥ +0.05 → хорошие; иначе плохие. Опора — прогоны
из tools/tuning/base_runs.json (зерно → номер). «Пауза» только останавливает прогон: пара стенда гасится, частичные итоги стираются,
карточка стоит в «Прогон»; метку сняли — прогон начинается заново с этапа 1 (автор 2026-11-21: «пауза служит только чтобы
останавливать прогон… продолжение всегда начинает прогон с этапа Прогон заново»).
Итог пары — в текст карточки (блоки «состояние» и таблица) и строкой в docs/TUNING.md. Принятое в опору не пишется само —
это решение остаётся за Claude и автором (карточка в «Хорошие» с пометкой).
Карточка-опора: в блоке «прогон» "seeds": [4, 5] и "base": true — одна пара текущей опоры (STAND/base.json) на этих зёрнах;
по итогу её номера записываются в base_runs.json (Δ в таблице карточки — к прежней опоре), карточка — в «Хорошие».
Автор 2026-11-21: «которые не ждут автора, должны автоматом гоняться, если не стоит пауза».

Дерево подбора (автор 2026-11-21: «после окончания прогона сразу бери следующий… текущий… эволюционирует (идёт в хорошие —
растёт ветка — новый узел в продолжении направления) либо отсеивается (идёт в плохие) и может порождать новый тикет (новая
ветка из текущего узла)»): итог пары — и сразу следующая карточка. «Хорошие» — изменения карточки пишутся в опору
(base.json стенда и tools/tuning/base.json, номера — в base_runs.json; зёрна, которых не было, — карточкой-опорой вперёд
очереди), и от узла рождается продолжение тем же шагом (числовые настройки: a → b даёт b → 2b − a). «Плохие» — от опоры
рождается новая ветка: польза была (Δ > 0) — полшага, вред (Δ ≤ 0) — шаг в другую сторону; от авто-ветки «полшага / обратно»
ветки не рождаются. Порядок очереди: карточки-опоры, потом продолжения, потом по доске сверху вниз.
Карты прогонов (автор: «показывай картинку в теле карточки») выкладываются в ветку board-maps репозитория и вставляются
в текст карточки картинками.
"""
import base64
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
# Стенд — в репозитории (tools/tuning/stand: скрипты, опора, журнал прогонов) либо DAMP_STAND из окружения / .env
# (docs/HARNESS.md). До 2026-11-21 стенд жил во временной папке прошлой сессии.
STAND = os.environ.get("DAMP_STAND") or B.CONF.get("DAMP_STAND") or os.path.join(HERE, "stand")
LOG = os.path.join(STAND, "board_runner.log")
TODO, QUEUE, RUN, ANALYSIS, GOOD, BAD = "Todo", "Очередь", "Прогон", "Анализ", "Хорошие", "Плохие"
STAGES = [[1, 3], [0, 2], [4, 5], [6, 7], [8, 9], [10, 11]]  # пары зёрен; решение — на 2, 6 и 12 зёрнах (CHECKPOINTS)
CHECKPOINTS = {1: (-0.01, 0.04), 3: (0.015, 0.05), 6: (0.025, 0.045)}  # этап: (порог Δ М1u, порог Δ слабых) — правило v3.3
SPEC = re.compile(r"```прогон\s*\n(.*?)\n```", re.S)
STATE = re.compile(r"\n*```состояние\s*\n(.*?)\n```", re.S)
RESULT = re.compile(r"\n*### Итоги прогонов.*", re.S)
REPO, MAPS_BRANCH = "qwertukg/DAMP", "board-maps"
UPLOADED = os.path.join(STAND, "maps_uploaded.json")
ATTRIBUTION = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def rest(method: str, path: str, data: dict | None = None) -> dict | None:
    """GitHub REST; 404 — None."""
    request = urllib.request.Request(
        "https://api.github.com" + path, method=method, data=None if data is None else json.dumps(data).encode(),
        headers={"Authorization": f"bearer {B.ISSUES_TOKEN if '/labels' in path or '/issues/' in path else B.TOKEN}",
                 "Accept": "application/vnd.github+json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


CARD_TYPES = ("DraftIssue", "Issue")  # карточки доски — задачи репозитория (автор: «сделай все issue, чтобы я мог комментарии добавлять»)
REPO_NAME = REPO.split("/")[1]
ISSUE_LABEL = "подбор"
_REPO_ID: list[str] = []


def repo_id() -> str:
    if not _REPO_ID:
        _REPO_ID.append(B.graphql("""query($o: String!, $n: String!) { repository(owner: $o, name: $n) { id } }""",
                                  {"o": B.OWNER, "n": REPO_NAME})["repository"]["id"])
    return _REPO_ID[0]


def save_card(content: dict, title: str | None = None, body: str | None = None) -> None:
    """Заголовок и / или текст карточки: задача репозитория (Issue) или черновик доски."""
    fields = {k: v for k, v in (("title", title), ("body", body)) if v is not None}
    if not fields:
        return
    names = ", ".join(f"{k}: ${k}" for k in fields)
    params = ", ".join(f"${k}: String!" for k in fields)
    if content.get("__typename") == "Issue":
        B.graphql(f"""mutation($i: ID!, {params}) {{ updateIssue(input: {{id: $i, {names}}}) {{ issue {{ id }} }} }}""",
                  {"i": content["id"], **fields})
    else:
        B.graphql(f"""mutation($i: ID!, {params}) {{ updateProjectV2DraftIssue(input: {{draftIssueId: $i, {names}}}) {{ draftIssue {{ id }} }} }}""",
                  {"i": content["id"], **fields})
    content.update(fields)


def to_issue(item_id: str) -> dict:
    """Черновик доски → задача репозитория DAMP с меткой «подбор»; вернуть карточку с новым содержимым.
    Токены раздельные (доска — только project, задачи — только Issues): задача создаётся токеном задач, на доску
    добавляется токеном доски с той же колонкой, черновик удаляется; один общий токен — прямое преобразование."""
    if B.ISSUES_TOKEN != B.TOKEN:
        draft = B.graphql("""query($i: ID!) { node(id: $i) { ... on ProjectV2Item { project { id } content { ... on DraftIssue { title body } }
            fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } } } } }""", {"i": item_id})["node"]
        created = B.graphql("""mutation($r: ID!, $t: String!, $b: String) { createIssue(input: {repositoryId: $r, title: $t, body: $b}) {
            issue { id number title body state } } }""", {"r": repo_id(), "t": draft["content"]["title"], "b": draft["content"]["body"]})["createIssue"]["issue"]
        added = B.graphql("""mutation($p: ID!, $c: ID!) { addProjectV2ItemById(input: {projectId: $p, contentId: $c}) { item { id } } }""",
                          {"p": draft["project"]["id"], "c": created["id"]})["addProjectV2ItemById"]["item"]
        found = {"id": added["id"], "content": {"__typename": "Issue", **created}, "fieldValues": {"nodes": []}}
        column = (draft.get("fieldValueByName") or {}).get("name")
        if column:
            move(B.board(), found, column)
        B.graphql("""mutation($p: ID!, $i: ID!) { deleteProjectV2Item(input: {projectId: $p, itemId: $i}) { deletedItemId } }""",
                  {"p": draft["project"]["id"], "i": item_id})
        try:
            rest("POST", f"/repos/{REPO}/issues/{created['number']}/labels", {"labels": [ISSUE_LABEL]})
        except Exception as error:  # noqa: BLE001
            say(f"метка задачи #{created['number']} не поставлена: {error!r}")
        return found
    found = B.graphql("""mutation($i: ID!, $r: ID!) { convertProjectV2DraftIssueItemToIssue(input: {itemId: $i, repositoryId: $r}) {
        item { id content { __typename ... on Issue { id number title body state } } } } }""",
                      {"i": item_id, "r": repo_id()})["convertProjectV2DraftIssueItemToIssue"]["item"]
    number = found["content"]["number"]
    try:
        rest("POST", f"/repos/{REPO}/issues/{number}/labels", {"labels": [ISSUE_LABEL]})
    except Exception as error:  # noqa: BLE001  метка — не главное
        say(f"метка задачи #{number} не поставлена: {error!r}")
    found["fieldValues"] = {"nodes": []}
    return found


def ensure_label() -> None:
    if rest("GET", f"/repos/{REPO}/labels/{urllib.request.quote(ISSUE_LABEL)}") is None:
        rest("POST", f"/repos/{REPO}/labels", {"name": ISSUE_LABEL, "color": "5319e7", "description": "Карточки доски «Подбор параметров DAMP»"})


def sync_issue_state(item: dict, column: str | None) -> None:
    """Состояние задачи не трогать: у доски включена встроенная автоматизация «Item closed» — закрытая задача уезжает в
    «Плохие» (2026-11-21 так съехали 13 «Хороших»); «Плохие» закрывает сама доска («Auto-close issue»)."""
    return


def maps_branch() -> None:
    """Пустая ветка board-maps (без истории проекта) — только карты прогонов для карточек доски."""
    if rest("GET", f"/repos/{REPO}/git/ref/heads/{MAPS_BRANCH}"):
        return
    blob = rest("POST", f"/repos/{REPO}/git/blobs", {"encoding": "utf-8", "content":
                "# Карты прогонов\n\nКарты прогонов стенда подбора (tune2-N.png) — картинки для карточек доски «Подбор параметров».\n"})
    tree = rest("POST", f"/repos/{REPO}/git/trees", {"tree": [{"path": "README.md", "mode": "100644", "type": "blob", "sha": blob["sha"]}]})
    commit = rest("POST", f"/repos/{REPO}/git/commits", {"message": "board-maps: карты прогонов для доски" + ATTRIBUTION,
                                                          "tree": tree["sha"], "parents": []})
    rest("POST", f"/repos/{REPO}/git/refs", {"ref": f"refs/heads/{MAPS_BRANCH}", "sha": commit["sha"]})


def map_url(number: int) -> str | None:
    """Ссылка на карту прогона в ветке board-maps; нет там — выложить data/maps/tune2-N.png; нет файла — None."""
    url = f"https://raw.githubusercontent.com/{REPO}/{MAPS_BRANCH}/maps/tune2-{number}.png"
    done = set(json.load(open(UPLOADED, encoding="utf-8"))) if os.path.exists(UPLOADED) else set()
    if number in done:
        return url
    # Выкладывание карт в ветку board-maps — это запись в репозиторий; включено DAMP_MAPS_UPLOAD=1 в .env (автор 2026-11-21:
    # «делай токен на запись обратно» — после того как карты пропали из комментариев).
    if not (os.environ.get("DAMP_MAPS_UPLOAD") or B.CONF.get("DAMP_MAPS_UPLOAD")):
        return None
    path = os.path.join(ROOT, "data", "maps", f"tune2-{number}.png")
    if not os.path.exists(path):
        return None
    maps_branch()
    target = f"/repos/{REPO}/contents/maps/tune2-{number}.png"
    found = rest("GET", f"{target}?ref={MAPS_BRANCH}")
    data = {"message": f"карта прогона {number}" + ATTRIBUTION, "branch": MAPS_BRANCH,
            "content": base64.b64encode(open(path, "rb").read()).decode()}
    if found:
        data["sha"] = found["sha"]
    rest("PUT", target, data)
    done.add(number)
    json.dump(sorted(done), open(UPLOADED, "w", encoding="utf-8"))
    return url


def maps_block(runs: dict, rows: dict) -> list[str]:
    """Раздел «Карты прогонов»: по картинке на законченный прогон карточки (зерно → номер)."""
    lines = []
    for seed, number in runs.items():
        if int(number) not in rows:
            continue
        try:
            url = map_url(int(number))
        except Exception as error:  # noqa: BLE001  сеть — карта будет при следующей записи
            say(f"карта {number} не выложена: {error!r}")
            url = None
        if url:
            lines += [f"**{number}** (зерно {seed})", "", f"![карта прогона {number}]({url})", ""]
    return (["", "### Карты прогонов", ""] + lines) if lines else []


def config_value(key: str):
    """Значение настройки в опоре: base.json стенда, иначе client/config.toml."""
    base = json.load(open(os.path.join(STAND, "base.json"), encoding="utf-8-sig"))
    if key in base:
        return base[key]
    tables = tomllib.load(open(os.path.join(ROOT, "client", "config.toml"), "rb"))
    for table in tables.values():
        if isinstance(table, dict) and key in table:
            value = table[key]
            return None if value == "none" else value
    value = tables.get(key)
    return None if value == "none" else value


def next_value(key: str, before, tried, kind: str):
    """Следующее значение ветки: continue — тем же шагом дальше; half — полшага от опоры; back — шаг в другую сторону."""
    if key.endswith("_FROM") and isinstance(before, int) and isinstance(tried, int) and tried > 0:
        # «С уровня N» (#17, 2026-11-21): дальше — на уровень ниже (включить шире), обратно — выше; полшага нет.
        value = {"continue": tried - 1, "back": tried + 1, "half": None}[kind]
        return value if value is not None and 1 <= value <= 6 and value != before else None
    if isinstance(before, list) and isinstance(tried, list) and len(before) == len(tried):
        values = [next_value(key, a, b, kind) if a != b else a for a, b in zip(before, tried)]
        return None if any(v is None for v in values) else values
    if isinstance(before, bool) or isinstance(tried, bool) or not all(isinstance(v, (int, float)) for v in (before, tried)):
        return None
    step = tried - before
    value = {"continue": tried + step, "half": before + step / 2, "back": before - step}[kind]
    if isinstance(before, int) and isinstance(tried, int):
        if kind == "half" and abs(step) < 2:
            return None
        value = int(round(value))
    else:
        value = round(value, 4)
    if re.search(r"SIMILARITY|QUANTILE", key):
        value = min(1.0, max(0.0, value))
    if (before >= 0 and tried >= 0 and value < 0) or value in (before, tried):
        return None
    return value


def base_now() -> dict:
    return json.load(open(os.path.join(STAND, "base.json"), encoding="utf-8-sig"))


PRIORITY_EMOJI = {3: "🔥", 2: "⬆️", 1: "➡️", 0: "⬇️"}  # 🔥 — критичный: задачи автора (Todo)


def spec_priority(body: str) -> int:
    found = SPEC.search(body or "")
    try:
        return int(json.loads(found.group(1)).get("priority", 0)) if found else 0
    except ValueError:
        return 0


def item_priority(item: dict) -> int:
    """Приоритет карточки — только эмодзи в начале заголовка (автор 2026-11-21: «тэг (поле) приоритета убери… только
    в заголовке»): ⬆️ 2, ➡️ 1, ⬇️ 0 — автор меняет его прямо в названии; эмодзи нет — «priority» из блока «прогон»
    (2 — опоры, 1 — продолжения «дальше», 0 — остальные)."""
    found = PRIORITY_TITLE.match((item.get("content") or {}).get("title") or "")
    if found:
        return {emoji: level for level, emoji in PRIORITY_EMOJI.items()}[found.group(1)]
    return spec_priority((item.get("content") or {}).get("body"))


def order_queue(project: dict) -> None:
    """Колонки, заголовки и порядок.
    Todo (автор 2026-11-21: «туду — сюда то, что ждёт слово автора, или я создаю там таски, у созданных мной критичный
    приоритет»): метка «ждёт слова автора» — карточка в Todo; в Todo без метки: запускаемая (блок «прогон», не начатая) —
    в Очередь, иначе это задача автора — 🔥 в заголовок (и у задач-Issue), Claude берёт её первой.
    Заголовок (задача автора #5): «приоритет итог было ---> стало, что проверяем»: приоритет — у Todo, Очереди, Прогона,
    Анализа; ✅ — Хорошие, ❌ — Плохие. Тело старых карточек — в новый вид (migrate_card).
    Порядок колонок — порядок доски: пришедшая карточка встаёт первой, Очередь берётся сверху (автор 2026-11-22)."""
    status = B.status_field(project)["name"]
    rows = log_rows()
    for item in project["items"]["nodes"]:
        content, fields = item.get("content") or {}, item_fields(item)
        column, label, body = fields.get(status), fields.get("Метка"), content.get("body") or ""
        if label == "ждёт слова автора" and column == QUEUE:
            # Автор 2026-11-21: «моё „делай“ не нужно ждать, бери задачу сразу, как я её передвинул»: карточку с меткой
            # в Очередь кладёт только автор — это и есть согласие. Метку снять, карточка берётся по порядку Очереди.
            label_field = next(f for f in project["fields"]["nodes"] if f.get("name") == "Метка")
            B.graphql("""mutation($p: ID!, $i: ID!, $f: ID!) { clearProjectV2ItemFieldValue(input: {projectId: $p, itemId: $i,
                fieldId: $f}) { clientMutationId } }""", {"p": project["id"], "i": item["id"], "f": label_field["id"]})
            item["fieldValues"]["nodes"] = [v for v in item["fieldValues"]["nodes"] if not (v and v.get("field", {}).get("name") == "Метка")]
            comment(item, "🤖 Прогонщик: карточка в «Очереди» — беру без «делай»: метку «ждёт слова автора» снял."
                    + ("" if SPEC.search(body) else " Блока «прогон» нет — нужна правка кода, её делает Claude (или GLM, если Claude нет)."))
            say(f"автор передвинул в Очередь: {card_name(content.get('title') or '')} — метка снята")
        elif label == "ждёт слова автора" and column == ANALYSIS:
            move(project, item, TODO, "метка «ждёт слова автора» — карточку никто не берёт, пока автор не решит")
    for item in project["items"]["nodes"]:
        content, fields = item.get("content") or {}, item_fields(item)
        if content.get("__typename") not in CARD_TYPES:
            continue
        sync_issue_state(item, fields.get(status))
        try:
            sync_card_fields(project, item)  # «Версия» и «Прогоны» всегда актуальны (автор 2026-11-22)
        except Exception as error:  # noqa: BLE001
            say(f"поля карточки не обновлены: {error!r}")
        own = bool(CARD.search(content.get("body") or ""))  # карточка прогонщика; без блока «карточка» — задача автора
        if not own and content.get("__typename") == "Issue" and SPEC.search(content.get("body") or ""):
            migrate_card(item, rows)  # карточка прогонщика, созданная сразу задачей (add_card): в обычный вид
            own = True
        body0 = content.get("body") or ""
        if content.get("__typename") == "DraftIssue" and not SPEC.search(body0) and not CARD.search(body0):
            # Черновик автора. Todo — его кухня, не трогать (задача #63: «свои задачи я буду сам ставить в очередь.
            # срабатывать только на очередь»); в Очереди — сделать задачей репозитория (комментарии), без переписывания.
            if fields.get(status) != QUEUE:
                continue
            converted = to_issue(item["id"])
            item["content"] = content = converted["content"]
            say(f"задача автора в Очереди: #{content.get('number')} {content.get('title')}")
            own = False
        elif content.get("__typename") == "DraftIssue":
            migrate_card(item, rows)
            to_issue(item["id"])  # и карточки прогонщика — задачами (комментарии)
            own = True
        column, label, body = fields.get(status), fields.get("Метка"), content.get("body") or ""
        runnable = bool(SPEC.search(body))
        held = PRIORITY_TITLE.match(content["title"])
        # Приоритет — всегда и первым (автор 2026-11-21: «показывай приоритет всегда. первым»): поставленный в заголовке
        # (автором или прогонщиком), у задачи автора в Todo — 🔥, иначе — «priority» блока «прогон» (по умолчанию ⬇️).
        if held:
            level = {e: lv for lv, e in PRIORITY_EMOJI.items()}[held.group(1)]
        elif not own and column == TODO:
            continue  # задача автора в Todo — не трогать (#63)
        elif not own and column == QUEUE:
            level = 3  # задача автора в Очереди — 🔥
        else:
            level = item_priority(item)
        marks = [PRIORITY_EMOJI[level]]
        marks += ["✅"] if column == GOOD else ["❌"] if column == BAD else []
        seed_state = card_state(item) or {}
        tail = f" · зерно {seed_state['seed']}" if "seed" in seed_state else ""
        core = title_core(card_meta(item)) + tail if own else TITLE_MARKS.sub("", content["title"])  # задачу автора не переписывать
        title = " ".join(marks + [core])
        if title != content["title"]:
            save_card(content, title=title)
            if column == TODO and level == 3 and not held:
                say(f"задача автора в Todo: {title}")
    # Порядок колонок (автор 2026-11-22: «задачи, когда в новую колонку переезжают, — всегда сверху» и «разбираться
    # колонка тоже должна сверху»): порядок доски — главный. Пришедшая в колонку карточка встаёт первой (move() → to_top,
    # перетащенные руками и созданные прямо в колонке — здесь); остальное прогонщик не пересортировывает, ручной порядок
    # автора сохраняется. Очередь берётся строго сверху (seed_launch, step). Прежняя сортировка по приоритету (#60, #66)
    # снята: эмодзи приоритета в заголовке — для глаз и для новых узлов дерева.
    _, moved = arrival_times(project, status)
    for item in reversed(moved):  # каждый подъём ставит карточку первой: снизу вверх — верхний из пришедших остаётся верхним
        try:
            B.graphql("""mutation($p: ID!, $i: ID!) { updateProjectV2ItemPosition(input: {projectId: $p, itemId: $i}) {
                items(first: 1) { nodes { id } } } }""", {"p": project["id"], "i": item["id"]})
        except Exception as error:  # noqa: BLE001
            say(f"карточка не поднята наверх: {error!r}")


SEEN = os.path.join(STAND, "comments_seen.json")
CLAUDE_MARK = "🤖 Claude:"  # так начинаются комментарии Claude (токен у Claude и автора один — по логину не различить)


def check_comments(project: dict) -> None:
    """Комментарии к задачам доски (автор 2026-11-21: «сделай, чтобы реагировала на комментарии»): новый комментарий не от
    Claude — задачу открыть заново, карточку в Todo (🔥 ставит order_queue), строка «комментарий автора» в журнал —
    по ней Claude берётся за ответ. У черновиков доски комментариев нет — только у задач (Issue)."""
    items = B.items("id content { ... on Issue { id number title state body comments(last: 30) { nodes { id body } } } }")
    first = not os.path.exists(SEEN)
    seen = set(json.load(open(SEEN, encoding="utf-8"))) if not first else set()
    status = B.status_field(project)["name"]
    columns = {i["id"]: item_fields(i).get(status) for i in project["items"]["nodes"]}
    for item in items:
        issue = item.get("content") or {}
        for comment in (issue.get("comments") or {}).get("nodes", []):
            if comment["id"] in seen:
                continue
            seen.add(comment["id"])
            if first or comment["body"].lstrip().startswith("🤖"):
                continue
            text = " ".join(comment["body"].split())[:300]
            if columns.get(item["id"]) == TODO:
                continue  # комментарии ловятся везде, кроме Todo (автор 2026-11-21); задачи — только из Очереди (#63)
            if CARD.search(issue.get("body") or ""):
                # Карточка прогона: колонку не трогать (прогон / решение идут своим ходом) — Claude отвечает в этой задаче.
                say(f"комментарий автора: #{issue['number']} «{issue['title']}» (карточка прогона): {text}")
                continue
            say(f"комментарий автора: #{issue['number']} «{issue['title']}»: {text}")
    json.dump(sorted(seen), open(SEEN, "w", encoding="utf-8"))


def add_card(project: dict, title: str, body: str) -> None:
    created = B.graphql("""mutation($p: ID!, $t: String!, $b: String) {
        addProjectV2DraftIssue(input: {projectId: $p, title: $t, body: $b}) { projectItem { id } } }""",
                        {"p": project["id"], "t": f"{PRIORITY_EMOJI[spec_priority(body)]} {title}", "b": body})
    item = to_issue(created["addProjectV2DraftIssue"]["projectItem"]["id"])
    move(project, item, QUEUE)


PLAN_TABLE = os.path.join(HERE, "plan.md")
PLAN_SEEN = os.path.join(STAND, "plan_seen.json")
PLAN_ROW = re.compile(r"^\|\s*(v\d+\.\d+)\s*\|")
PLAN_MARK = "<!-- план: {} -->"


def plan_rows() -> list[dict]:
    """Строки таблицы плана tools/tuning/plan.md (задача автора #88: «пусть листенер слушает таблицу с планом тоже, и если
    появились новые задачи, берёт их в работу в канбан»). Колонки: № | Приоритет | Было | Стало | Что пробуем |
    Настройки (JSON без «|», пусто или «код» — нужна правка кода) | Подробно."""
    if not os.path.exists(PLAN_TABLE):
        return []
    rows = []
    for line in open(PLAN_TABLE, encoding="utf-8"):
        found = PLAN_ROW.match(line)
        if not found:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        number, priority, was, became, short_text, settings, full = cells[:7]
        settings = settings.strip("` ")
        rows.append({"number": number, "priority": priority, "was": was.strip("[]"), "became": became.strip("[]"),
                     "short": short_text, "settings": settings, "full": full})
    return rows


def plan_sync(project: dict) -> None:
    """Новая строка таблицы плана — карточка в «Очередь» (один раз на номер; уже есть на доске — не повторять)."""
    rows = plan_rows()
    if not rows:
        return
    seen = json.load(open(PLAN_SEEN, encoding="utf-8")) if os.path.exists(PLAN_SEEN) else {}
    on_board = {m for item in project["items"]["nodes"]
                for m in re.findall(r"<!-- план: (v\d+\.\d+) -->", (item.get("content") or {}).get("body") or "")}
    level = {emoji: lvl for lvl, emoji in PRIORITY_EMOJI.items()}
    # Каждая новая карточка встаёт наверх Очереди (move → to_top), а Очередь берётся сверху: новые строки — с конца
    # таблицы, чтобы первая строка плана оказалась верхней и пошла первой (2026-11-23: v5.5 ⬇️ встала первой).
    for row in reversed(rows):
        if row["number"] in seen or row["number"] in on_board:
            continue
        priority = level.get(row["priority"], 1)
        spec_text = None
        if row["settings"] and row["settings"].lower() not in ("код", "—", "-"):
            try:
                changes = json.loads(row["settings"])
            except ValueError as error:
                say(f"план {row['number']}: настройки не JSON — {error}; строка пропущена")
                seen[row["number"]] = "ошибка"
                continue
            spec_text = json.dumps({"changes": changes, "label": row["short"], "priority": priority, "plan": row["number"]},
                                   ensure_ascii=False)
        meta = {"short": row["short"], "full": row["full"] + ("" if spec_text else " Нужна правка кода — делает Claude (без Claude — GLM)."),
                "was": row["was"] or "—", "became": row["became"] or "—", "manual": True}
        body = compose(meta, spec_text, None, [], "", "") + "\n\n" + PLAN_MARK.format(row["number"])
        title = f"{PRIORITY_EMOJI[priority]} {boxed(meta['was'])} → {boxed(meta['became'])} {row['short']}"
        created = B.graphql("""mutation($p: ID!, $t: String!, $b: String) {
            addProjectV2DraftIssue(input: {projectId: $p, title: $t, body: $b}) { projectItem { id } } }""",
                            {"p": project["id"], "t": title, "b": body})
        item = to_issue(created["addProjectV2DraftIssue"]["projectItem"]["id"])
        move(project, item, QUEUE, f"новая строка {row['number']} таблицы плана (tools/tuning/plan.md)"
             + ("" if spec_text else " — блока «прогон» нет, нужна правка кода"))
        seen[row["number"]] = item["id"]
        json.dump(seen, open(PLAN_SEEN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        say(f"план: {row['number']} → карточка в Очереди — {row['short']}")


def same_spec_waiting(project: dict, changes: dict) -> bool:
    """Такая же карточка уже есть на той же опоре (или ждёт в очереди) — не плодить."""
    key, base = json.dumps(changes, sort_keys=True), base_now()
    for item in project["items"]["nodes"]:
        body = (item.get("content") or {}).get("body") or ""
        spec, state = SPEC.search(body), STATE.search(body)
        if not spec or json.dumps(json.loads(spec.group(1)).get("changes", {}), sort_keys=True) != key:
            continue
        if not state or json.loads(state.group(1)).get("base") == base:
            return True
    return False


def short(key: str) -> str:
    return key.replace("LADDER_", "").replace("ACTIVATION_", "").lower()


def tested(project: dict, changes: dict) -> bool:
    """Значение уже проверено (на любой опоре): есть карточка с теми же изменениями или оно в списке «tested» дерева
    (проверенные до доски, напр. поправка радиуса 0 = Б2, 4 и 6 уровней, слияние 0.8)."""
    key = json.dumps(changes, sort_keys=True)
    for item in project["items"]["nodes"]:
        found = SPEC.search((item.get("content") or {}).get("body") or "")
        if found and json.dumps(json.loads(found.group(1)).get("changes", {}), sort_keys=True) == key:
            return True
    done = json.load(open(TREE_PATH, encoding="utf-8")).get("tested", {})
    return len(changes) == 1 and any(json.dumps(v) == json.dumps(value) for name, value in changes.items() for v in done.get(name, []))


def grow(project: dict, item: dict, state: dict, outcome: str, mean: float) -> None:
    """Узел дерева: хороший — продолжение тем же шагом; плохой — полшага (Δ > 0) или обратно (Δ ≤ 0) от опоры."""
    content = item["content"]
    spec = json.loads(SPEC.search(content["body"]).group(1))
    if spec.get("base") or (outcome == "bad" and spec.get("gen") in ("half", "back")):
        return
    # Плато (разбор методики 2026-11-21): «полшага» / «обратно» — только при заметном эффекте, |Δ| ≥ 0.03 по 4+ зёрнам;
    # иначе ветка — плато, узлов не порождать (5 авто-узлов при |Δ| ≤ 0.035 не дали ни одного принятия).
    if outcome == "bad" and spec.get("gen") != "continue" and (len(state.get("runs", {})) < 4 or abs(mean) < 0.03):
        return
    # Продолжение, которое не удалось, назад не разворачивается (там — прежняя, худшая опора): только полшага.
    kind = "continue" if outcome == "good" else ("half" if mean > 0 or spec.get("gen") == "continue" else "back")
    before_all = state.get("before") or {}
    changes, parts = {}, []
    for key, tried in spec.get("changes", {}).items():
        before = before_all.get(key, config_value(key))  # grow зовётся до accept: опора ещё прежняя
        value = next_value(key, before, tried, kind) if before is not None else None
        if value is None:
            return
        changes[key] = value
        start = tried if kind == "continue" else before
        parts.append(f"{short(key)} {json.dumps(start, ensure_ascii=False)} → {json.dumps(value, ensure_ascii=False)}")
    if not changes or same_spec_waiting(project, changes) or tested(project, changes):
        return
    word = {"continue": "дальше", "half": "полшага", "back": "обратно"}[kind]
    why = {"continue": f"родитель принят (Δ М1u {mean:+.3f}) и записан в опору — тот же шаг дальше",
           "half": f"родитель закрыт (Δ М1u {mean:+.3f}), но направление живое — полшага от опоры",
           "back": f"родитель закрыт, вред (Δ М1u {mean:+.3f}) — шаг от опоры в другую сторону"}[kind]
    label = f"{word}: " + ", ".join(parts)
    body = (f"**Что:** авто-узел дерева подбора от «{content['title']}»: " + ", ".join(f"`{k}` → {json.dumps(v, ensure_ascii=False)}"
            for k, v in changes.items()) + f".\n\n**Почему:** {why}.\n\n"
            + "```прогон\n" + json.dumps({"changes": changes, "label": label, "parent": content["title"], "parent_id": item["id"], "gen": kind,
                                         "priority": 1 if kind == "continue" else 0}, ensure_ascii=False) + "\n```")
    add_card(project, label, body)
    say(f"новый узел: {label} (от «{content['title']}»)")


def accept(project: dict, item: dict, state: dict) -> None:
    """Хороший узел — в опору: изменения в base.json (стенд и tools/tuning), номера — в base_runs.json; зёрна без
    прогонов этой опоры — карточкой-опорой вперёд очереди."""
    changes = json.loads(SPEC.search(item["content"]["body"]).group(1)).get("changes", {})
    for path in (os.path.join(STAND, "base.json"), os.path.join(HERE, "base.json")):
        base = json.load(open(path, encoding="utf-8-sig"))
        base.update(changes)
        json.dump(base, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    runs_path = os.path.join(HERE, "base_runs.json")
    # Новая опора — только прогоны принятой карточки (прежние зёрна опоры устарели); остальные из BASE_SEEDS — карточками-
    # опорами по паре вперёд очереди (правило v3.2: опора на 12 зёрнах).
    runs = {s: int(n) for s, n in state["runs"].items()}
    json.dump(runs, open(runs_path, "w", encoding="utf-8"))
    missing = [s for s in BASE_SEEDS if str(s) not in runs]
    title = card_name(item["content"]["title"])
    for i in range(0, len(missing), 2):
        pair = missing[i:i + 2]
        add_card(project, f"Опора на зёрнах {'/'.join(map(str, pair))} после «{title}»",
                 f"**Что:** новая опора после «{title}» на зёрнах {', '.join(map(str, pair))} — среднее опоры по 12 зёрнам (правило {RULE}).\n\n"
                 "```прогон\n" + json.dumps({"changes": {}, "label": "опора", "seeds": pair, "base": True, "priority": 2},
                                            ensure_ascii=False) + "\n```")
    say(f"в опору: {changes} — {item['content']['title']}")


def say(text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as out:
        out.write(line + "\n")


def log_rows() -> dict:
    rows = {}
    path = os.path.join(STAND, "tune2_log.jsonl")
    for line in open(path, encoding="utf-8"):
        entry = json.loads(line)
        rows[int(entry["run"])] = entry
    return rows


def m1u(entry: dict) -> float | None:
    values = entry.get("m1_digits_mean")
    return None if not values else sum(values) / len(values)


def next_number() -> int:
    used = [int(m.group(1)) for name in os.listdir(STAND) for m in [re.match(r"tune2-(\d+)\.json$", name)] if m]
    used += list(log_rows()) if os.path.exists(os.path.join(STAND, "tune2_log.jsonl")) else []  # новая машина: задания не в git
    number = max(used or [0]) + 1
    return number + (number % 2)


def launch(numbers: list[int], seeds: list[int], changes: dict, label: str) -> None:
    base = json.load(open(os.path.join(STAND, "base.json"), encoding="utf-8-sig"))
    for number, seed in zip(numbers, seeds):
        over = {**base, **changes}
        if seed:
            over["ENCODER_RANDOM_SEED"] = seed
        spec = {"label": f"250 цифр: {label}, зерно {seed}", "changes": over}
        json.dump(spec, open(os.path.join(STAND, f"tune2-{number}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    done = os.path.join(STAND, "tune_pair.done")
    if os.path.exists(done):
        os.remove(done)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    subprocess.Popen(["pwsh", "-NoProfile", "-File", os.path.join(STAND, "tune_pair.ps1"), "-List", ",".join(map(str, numbers)),
                      "-Count", "250", "-Tests", "100"], creationflags=flags, cwd=ROOT)
    say(f"запущено {numbers} (зёрна {seeds}) — {label}")


def finished(numbers: list[int]) -> bool:
    rows = log_rows()
    if all(int(n) in rows for n in numbers):  # итоги уже в журнале стенда (tune_pair.done мог стереть запуск следующей пары)
        return True
    done = os.path.join(STAND, "tune_pair.done")
    if not os.path.exists(done):
        return False
    text = open(done, encoding="utf-8-sig").read()
    return all(re.search(rf"(?m)^{n} ", text) for n in numbers)


def pair_busy() -> bool:
    """Идёт ли пара стенда (своя или запущенная вручную): tune_pair.done нет, а журналы пары свежие."""
    done = os.path.join(STAND, "tune_pair.done")
    if os.path.exists(done):
        return False
    outs = [os.path.join(STAND, n) for n in os.listdir(STAND) if re.match(r"tune2-\d+\.out$", n)]
    return bool(outs) and time.time() - max(os.path.getmtime(p) for p in outs) < 600


def stop(numbers: list[int]) -> None:
    """Погасить пару стенда: tune_pair.ps1 и tune_cycle.py этих номеров со всеми дочерними; отметить пару законченной."""
    pattern = "|".join(str(n) for n in numbers)
    script = (r"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'tune_(cycle\.py|pair\.ps1)' -and "
              + r"$_.CommandLine -match '\b(" + pattern + r")\b' } | ForEach-Object { taskkill /T /F /PID $_.ProcessId | Out-Null }")
    subprocess.run(["pwsh", "-NoProfile", "-Command", script], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with open(os.path.join(STAND, "tune_pair.done"), "a", encoding="utf-8") as out:
        for number in numbers:
            out.write(f"{number} stopped=пауза\n")


def clear(item: dict) -> None:
    """Стереть частичные итоги и состояние: следующий прогон карточки — заново (шапка, блок «прогон» и дерево остаются)."""
    body = item["content"].get("body") or ""
    spec_found, tree = SPEC.search(body), TREE_BLOCK.search(body)
    set_body(item, compose(card_meta(item), spec_found.group(1) if spec_found else None, None, [], "", tree.group(0) if tree else ""))


def item_fields(item: dict) -> dict:
    values = {}
    for value in item["fieldValues"]["nodes"]:
        if value and value.get("field"):
            values[value["field"]["name"]] = value.get("name") if "name" in value else value.get("text")
    return values


MOVE_WHY = {"Todo": "нужно слово автора", "Очередь": "в очередь", "Прогон": "прогон запущен",
            "Анализ": "прогон досчитан", "Хорошие": "правка принята", "Плохие": "правка закрыта"}


def stdout_block(lines) -> str:
    """stdout под катом в конце комментария (задача автора #105: «перед любым действием подготавливай stdout и… добавляй
    его в конец сообщения под кат»)."""
    text = "\n".join(str(line) for line in lines).replace("```", "ʼʼʼ").strip()
    if not text:
        return ""
    return "\n\n<details><summary>stdout</summary>\n\n```\n" + text[-3500:] + "\n```\n\n</details>"


def log_tail(path: str, count: int = 8) -> list[str]:
    try:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()[-count:]
    except OSError:
        return []


def comment(item: dict, text: str, stdout=None) -> None:
    """Комментарий в задаче-карточке (автор 2026-11-21: «каждое движение по доске сопровождай комментом… если ошибка —
    выводи в коммент»; #105: «в начале строки всегда начинай с того, кто к нам обращается, как в пьесе» — «🤖 Прогонщик:»,
    «🤖 Claude:», «🤖 GLM:», «🤖 Роутер:»; в конце — stdout под катом: по умолчанию хвост журнала прогонщика)."""
    content = item.get("content") or {}
    if content.get("__typename") != "Issue" or not content.get("id"):
        return
    body = text if text.startswith("🤖") else f"🤖 Прогонщик: {text}"
    body += stdout_block(log_tail(LOG) if stdout is None else stdout)
    try:
        B.graphql("""mutation($i: ID!, $b: String!) { addComment(input: {subjectId: $i, body: $b}) { clientMutationId } }""",
                  {"i": content["id"], "b": body})
    except Exception as error:  # noqa: BLE001  комментарий не должен ронять шаг
        say(f"комментарий не оставлен: {error!r}")


ARRIVALS = os.path.join(STAND, "arrivals.json")  # карточка → (колонка, когда в неё пришла): порядок «пришедшие — сверху»
ARRIVALS_BURST = 5  # больше «пришедших» за шаг мимо move() — считать потерей записей, а не приходом


def _arrivals() -> dict | None:
    """Записи прихода; None — файла нет или он испорчен (тогда запомнить заново, ничего не поднимая)."""
    try:
        return json.load(open(ARRIVALS, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_arrivals(data: dict) -> None:
    """Атомарно: прогонщик гасят принудительно, обрезанный файл давал «все карточки пришли» (2026-11-22)."""
    temporary = f"{ARRIVALS}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False)
    os.replace(temporary, ARRIVALS)


def to_top(project: dict, item: dict, column: str) -> None:
    """Автор 2026-11-22: «сделай, что задачи, когда в новую колонку переезжают, всегда сверху клались» — запомнить время
    прихода и сразу поставить карточку первой (в виде доски по колонкам первая в списке — верхняя в своей колонке)."""
    data = _arrivals() or {}
    data[item["id"]] = {"column": column, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _save_arrivals(data)
    try:
        B.graphql("""mutation($p: ID!, $i: ID!) { updateProjectV2ItemPosition(input: {projectId: $p, itemId: $i}) {
            items(first: 1) { nodes { id } } } }""", {"p": project["id"], "i": item["id"]})
    except Exception as error:  # noqa: BLE001  порядок не должен ронять шаг
        say(f"карточка не поднята наверх: {error!r}")


def arrival_times(project: dict, status: str) -> tuple[dict, list[dict]]:
    """Когда каждая карточка пришла в свою нынешнюю колонку. Новая в записи — время смены поля Status (updatedAt), без
    него — создание карточки; колонка сменилась мимо move() (автор перетащил руками) — «сейчас»."""
    data, changed = _arrivals(), False
    seeded = bool(data)  # первый запуск или испорченный файл — только запомнить, ничего не поднимать
    data = data or {}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    moved = []
    for item in project["items"]["nodes"]:
        column = item_fields(item).get(status)
        record = data.get(item["id"])
        if record is None:
            stamp = next((v.get("updatedAt") for v in (item.get("fieldValues") or {}).get("nodes", [])
                          if v and v.get("field", {}).get("name") == status and v.get("updatedAt")), None)
            data[item["id"]] = {"column": column, "at": now if seeded else (stamp or item.get("createdAt") or now)}
            changed = True
            if seeded:
                moved.append(item)  # новая карточка (автор создал прямо в колонке) — тоже наверх
        elif record.get("column") != column:
            data[item["id"]] = {"column": column, "at": now}
            changed = True
            moved.append(item)
    if changed:
        _save_arrivals(data)
    if len(moved) > ARRIVALS_BURST:
        # Столько карточек сразу мимо move() автор не двигает: это потеря записей, а не приход — порядок доски не трогать.
        say(f"порядок: {len(moved)} карточек «пришли» разом мимо move() — похоже на потерю записей, наверх не поднимаю")
        moved = []
    return {key: value["at"] for key, value in data.items()}, moved


def move(project: dict, item: dict, column: str, why: str | None = None) -> None:
    """Колонка карточки + комментарий «→ колонка: почему» (если колонка меняется)."""
    field = B.status_field(project)
    current = next((v.get("name") for v in (item.get("fieldValues") or {}).get("nodes", [])
                    if v and v.get("field", {}).get("name") == field["name"]), None)
    option = next(o["id"] for o in field["options"] if o["name"] == column)
    B.graphql("""mutation($p: ID!, $i: ID!, $f: ID!, $o: String!) { updateProjectV2ItemFieldValue(input: {projectId: $p, itemId: $i,
        fieldId: $f, value: {singleSelectOptionId: $o}}) { projectV2Item { id } } }""",
              {"p": project["id"], "i": item["id"], "f": field["id"], "o": option})
    if current != column:
        to_top(project, item, column)
        why = why or MOVE_WHY.get(column, "")
        comment(item, f"🤖 Прогонщик: → «{column}».{chr(10) * 2 if chr(10) in why else ' '}{why}".rstrip())
        nodes = item.setdefault("fieldValues", {"nodes": []})["nodes"]
        nodes[:] = [v for v in nodes if not (v and v.get("field", {}).get("name") == field["name"])]
        nodes.append({"name": column, "field": {"name": field["name"]}})


def run_report(item: dict, state: dict, rows: dict, next_text: str) -> str:
    """Комментарий-итог прогона (автор 2026-11-21): как стало — значение; что дальше; карта «как стало». «Как было» — в теле."""
    meta = card_meta(item, state)
    base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
    seed, run = str(state["seed"]), state.get("run")
    entry, ref_run = rows.get(int(run)) if run else None, base.get(seed)
    ref = rows.get(int(ref_run)) if ref_run else None
    became = meta.get("became")
    value = (f"{boxed(became)} — " if became and became != "—" else "опора — ") + f"{tag(PLAN, [run])}, зерно {seed}"
    if entry and m1u(entry) is not None:
        delta = f" (Δ {m1u(entry) - m1u(ref):+.3f} к опоре)" if ref and m1u(ref) is not None else ""
        value += (f": М1u {m1u(entry):.3f}{delta}, слабые 2, 4, 5 {weak(entry):.3f}, монопятен {entry.get('whole_mean')},"
                  f" честная {entry.get('top_honest')}")
    lines = [f"- **Как стало:** {value}", f"- **Что дальше:** {next_text}", map_line(item, run)]
    return "\n".join(lines)


MAPS_PENDING = os.path.join(STAND, "maps_pending.json")  # карточка → прогон, чья карта не выложилась: дослать комментарием


def map_line(item: dict, run, title: str = "Карта, как стало") -> str:
    """Строка с картой прогона для комментария (автор 2026-11-22: «после Анализа карта раскладки в комментарии должна быть
    обязательно!»): три попытки выложить; не вышло — карточка в maps_pending, карта досылается отдельным комментарием."""
    if not run:
        return f"- **{title}:** прогона нет"
    for attempt in range(3):
        url = None
        try:
            url = map_url(int(run))
        except Exception as error:  # noqa: BLE001
            say(f"карта {run} не выложена (попытка {attempt + 1}): {error!r}")
        if url:
            return f"- **{title}:**\n\n![карта {run}]({url})"
        time.sleep(5)
    pending = json.load(open(MAPS_PENDING, encoding="utf-8")) if os.path.exists(MAPS_PENDING) else {}
    pending[item["id"]] = int(run)
    json.dump(pending, open(MAPS_PENDING, "w", encoding="utf-8"))
    return f"- **{title}:** пока не выложена (`data/maps/tune2-{run}.png`) — пришлю отдельным комментарием"


def maps_resend(project: dict) -> None:
    """Дослать карты, которые не выложились при итоге (maps_pending)."""
    if not os.path.exists(MAPS_PENDING):
        return
    pending = json.load(open(MAPS_PENDING, encoding="utf-8"))
    by_id = {item["id"]: item for item in project["items"]["nodes"]}
    for item_id, run in list(pending.items()):
        try:
            url = map_url(int(run))
        except Exception as error:  # noqa: BLE001
            say(f"карта {run} снова не выложена: {error!r}")
            continue
        if url and item_id in by_id:
            comment(by_id[item_id], f"🤖 Прогонщик: карта прогона {tag(PLAN, [run])}, как стало:\n\n![карта {run}]({url})")
        if url or item_id not in by_id:
            pending.pop(item_id)
    json.dump(pending, open(MAPS_PENDING, "w", encoding="utf-8"))


PLAN = None  # версия плана — по номеру прогона (plan_versions.json) и заголовку plan.md, см. tag(), current_plan()
PLAN_TITLE = re.compile(r"^(v\d+) · ")
PREFIX = re.compile(r"^(\d+\.\d+\.\d+|v\d+)\+[\d–, ]+ · ")  # прежнее обозначение в заголовке — убрать


def plan_version(title: str) -> str:
    """Версия плана карточки: «v3 · …» в заголовке, иначе текущая (plan.md)."""
    found = PLAN_TITLE.match(PREFIX.sub("", title or ""))
    return found.group(1) if found else current_plan()


def span(numbers) -> str:
    """Номера прогонов коротко: 720, 721, 722, 723, 730 → «720–723, 730»."""
    ordered, parts, i = sorted({int(n) for n in numbers}), [], 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(str(ordered[i]) if i == j else f"{ordered[i]}–{ordered[j]}")
        i = j + 1
    return ", ".join(parts)


PLAN_VERSIONS = os.path.join(HERE, "plan_versions.json")  # версия плана → номер первого её прогона


def plan_starts() -> dict:
    try:
        return {k: int(v) for k, v in json.load(open(PLAN_VERSIONS, encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {"v2": 574, "v3": 660}


def current_plan() -> str:
    """Версия плана сейчас — из заголовка tools/tuning/plan.md («# План подбора v4 …»); нет — последняя известная."""
    try:
        found = re.search(r"План подбора (v\d+)", open(PLAN_TABLE, encoding="utf-8").readline())
        if found:
            return found.group(1)
    except OSError:
        pass
    return max(plan_starts().items(), key=lambda pair: pair[1])[0]


def ensure_plan_start() -> None:
    """Новая версия плана в plan.md — запомнить, с какого прогона она идёт (следующий номер стенда)."""
    starts, plan = plan_starts(), current_plan()
    if plan not in starts:
        starts[plan] = next_number()
        json.dump(starts, open(PLAN_VERSIONS, "w", encoding="utf-8"), ensure_ascii=False)
        say(f"версия плана {plan} — с прогона {starts[plan]}")


def version_of_run(number) -> str:
    starts = sorted(plan_starts().items(), key=lambda pair: pair[1])
    found = starts[0][0]
    for name, first in starts:
        if int(number) >= first:
            found = name
    return found


def tag(plan: str | None, numbers) -> str:
    """Обозначение прогона «версия плана . номера» (автор 2026-11-21: «пиши версию и номер прогона… версия плана я имел
    в виду… (<v3>.<660>)»): v3.720–723. Версия — по номеру прогона (plan_versions.json), а не по константе: автор
    2026-11-22 «тег версии… должен быть актуальным». В заголовок карточки не пишется — только поля, тексты и дерево."""
    groups: dict = {}
    for number in numbers:
        groups.setdefault(version_of_run(number), []).append(int(number))
    if not groups:
        return f"{plan or current_plan()}.—"
    return ", ".join(f"{name}.{span(runs)}" for name, runs in sorted(groups.items(), key=lambda pair: min(pair[1])))


TREE_PATH = os.path.join(HERE, "board_tree.json")
TREE_BLOCK = re.compile(r"\n*### Дерево прогонов\n(```text\n.*?\n```|<pre>.*?</pre>)", re.S)
MARKS = {"good": ".✅", "bad": ".❌", "run": ".⏳", "base": ""}


PRIORITY_TITLE = re.compile(r"^(🔥|⬆️|➡️|⬇️) ")
TITLE_MARKS = re.compile(r"^(?:(?:🔥|⬆️|➡️|⬇️|✅|❌|⏳)\s*)+")
CARD = re.compile(r"```карточка\s*\n(.*?)\n```", re.S)
SERVICE = re.compile(r"\n*<details><summary>Служебное</summary>.*?</details>", re.S)
MAPS = re.compile(r"\n*### Карты прогонов\n.*?(?=\n<details><summary>Служебное|\n### Дерево прогонов|\Z)", re.S)


def card_name(title: str) -> str:
    """Имя карточки без служебного начала: эмодзи приоритета и итога, «v3 · », старое обозначение прогона."""
    return re.sub(r" · зерно \d+$", "", PLAN_TITLE.sub("", PREFIX.sub("", TITLE_MARKS.sub("", title or ""))))


# Простые имена настроек для заголовка «было ---> стало, что проверяем» (задача автора #5, 2026-11-21).
SIMPLE = {
    "LADDER_MERGE_SIMILARITY": "слияние похожих окон", "LADDER_LAYOUT_LAMBDA_QUANTILE": "порог λ раскладки",
    "LADDER_CLUSTER_MIN_POINTS": "соседей в окнах уровней", "LADDER_TARGET_DENSITY": "плотность кода детали",
    "ACTIVATION_DETECTOR_FILL_CELL": "размер детекторов", "LADDER_CLUSTER_LEVELS": "уровней",
    "LADDER_KEYHOLE_RADIUS_SHARE": "размер окон по уровням", "LADDER_HIERARCHY_LAYERS": "слоёв детекторов",
    "ACTIVATION_SATURATION_LIMIT": "σ сильнейших детекторов", "LADDER_EMBEDDING_LAYERS": "слоёв в коде детали",
}
# Карточки, у которых нет блока «прогон» или изменения — не числа: (что проверяем, было, стало).
CARD_MANUAL = {
    "Разбор: шестёрка 648 целиком в одном пятне": ("разбор: почему шестёрка собирается целиком", "—", "—"),
    "Диагноз «код или раскладка» по цифрам": ("разбор: какой цифре мешает код, а какой раскладка", "—", "—"),
    "Мерка М1u по 3 раскладкам верха": ("главная мерка — доля точек цифры в её главном пятне", "М1ц", "М1u"),
    "Опора: 4 уровня — М1u 0.505": ("опора: 4 уровня, М1u 0.505", "—", "—"),
    "Б9: 5 уровней": ("уровней", "4", "5"),
    "Слияние окон мягче: 0.8 → 0.9": ("слияние похожих окон", "0.8", "0.9"),
    "А1: верх в режиме статьи": ("верхнюю карту раскладывать как в статье", "сглаживание", "режим статьи"),
    "А4: верх без отжига": ("отжиг на верхней карте", "есть", "нет"),
    "А2: η = ∞ на верхе (стенд верха)": ("крутизна порога λ на верхней карте", "η 10", "η ∞"),
    "Б1: плотность эмбеддинга 0.07": ("плотность кода детали", "0.1", "0.07"),
    "Б2: детекторы мельче (поправка радиуса 0)": ("детекторы мельче", "0.5", "0"),
    "Б2′: детекторы крупнее (поправка 1.0)": ("детекторы крупнее", "0.5", "1.0"),
    "Б3: одно окно на картинку наверху": ("одно окно на картинку на верхнем уровне", "много окон", "1 окно"),
    "Б4: грубое связывание наверху": ("грубое связывание места на верхнем уровне", "все слои места", "слои 0–6"),
    "Б6: σ 10 → 8": ("сколько сильнейших детекторов идёт в код", "σ 10", "σ 8"),
    "Б7: код детали без самого узкого слоя": ("код детали без самого узкого слоя", "5 слоёв", "4 слоя"),
    "Б8: уровень 3 видит части цифры": ("уровень 3 видит части цифры", "[1.15, 0.5, 2.0]", "[1.15, 0.5, 1.2, 2.0]"),
    "6 уровней": ("уровней", "5", "6"),
    "Окна уровня 2 крупнее": ("окна уровня 2 крупнее", "[10, 3, 14]", "[10, 5, 14]"),
    "Сочетание трёх «почти»": ("три «почти» вместе: части на ур. 3, плотность 0.07, детекторы крупнее", "по отдельности", "вместе"),
    "Структурные рычаги для 2, 4, 5": ("разбор: почему рвутся 2, 4, 5", "—", "—"),
    "Методика: «почти» — сразу на 6 зёрнах": ("правило отбора: сравнение со средним опоры, 6 зёрен", "v3.1", "v3.2"),
    "Записать опору (5 уровней, слияние 0.9) в config.toml": ("записать опору в config.toml", "12 уровней, слияние 0.8", "5 уровней, слияние 0.9"),
    "Стенд на 500 цифрах": ("учить стенд на 500 цифрах", "250", "500"),
    "Проверка на другом срезе MNIST": ("проверить опору на других картинках", "картинки 0–250", "случайные 250"),
    "Б5: плотность детали по уровням": ("плотность кода по уровням", "0.1 везде", "реже на уровнях 3–4"),
    "Б7′: σ по цвету λ_d — широкие слои первыми (код)": ("в код детали — сначала широкие слои детекторов", "по силе", "по цвету"),
    "Перечисляющий пулинг на уровнях 4–5 (код)": ("в код окна — все детали, а не одна сильнейшая", "сильнейшая", "все по очереди"),
    "Р2: смешанное ядро на уровнях 3–5 (код)": ("раскладка верха тянет далёкие куски класса", "гауссово ядро", "гауссово + ядро статьи"),
    "А3 + Р5: растворение и лучшая из K раскладок (код)": ("растворить карту и взять лучшую из K раскладок", "1 раскладка", "K раскладок"),
    "Р1: ядро статьи на уровнях 4–5 без роста λ": ("раскладка верха (ур. 4–5) по статье: тянет далёкие куски", "сглаживание", "ядро статьи"),
    "Р1′: ядро статьи с уровня 3": ("раскладка верха (ур. 3–5) по статье: тянет далёкие куски", "сглаживание", "ядро статьи"),
    "Р3: λ раскладки 0.9 отдельно от детекторов": ("порог λ только для раскладки верха", "λ 0.82", "λ 0.9"),
    "Плоскость места верха шире: окно уровня 5 видит 3 цифры": ("плоскость места верхнего окна шире", "[1.15, 0.5, 2.0]", "[1.15, 0.5, 2.0, 3.0]"),
}


def fmt(value) -> str:
    if value is None:
        return "нет"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{short(k)} {fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def derive_meta(name: str, body: str, spec: dict | None, state: dict | None) -> dict:
    """Что пробуем (коротко), подробно, было, стало — из старого текста карточки, блока «прогон» и состояния."""
    text = body or ""
    for pattern in (TREE_BLOCK, SERVICE, MAPS, SPEC, STATE, RESULT):
        text = pattern.sub("", text)
    full = re.sub(r"^\*\*Что:\*\*\s*", "", text.strip()).strip() or name
    spec = spec or {}
    changes = spec.get("changes", {})
    if name in CARD_MANUAL:
        brief, was, became = CARD_MANUAL[name]
    elif spec.get("base"):
        brief, was, became = f"опора на зёрнах {'/'.join(map(str, spec.get('seeds', [])))}", "—", "—"
    elif changes:
        brief, was, became = spec.get("short") or ", ".join(SIMPLE.get(k, short(k)) for k in changes), "", ""
    else:
        brief, was, became = name, "—", "—"
    return {"short": brief, "full": full, "was": was, "became": became, "manual": name in CARD_MANUAL}


def card_meta(item: dict, state: dict | None = None) -> dict:
    """Шапка карточки (задача автора #5): из блока «карточка», иначе из старого текста; «было / стало» числовых правок —
    из состояния (значение опоры на запуске) или нынешней опоры."""
    body = (item.get("content") or {}).get("body") or ""
    found, spec_found = CARD.search(body), SPEC.search(body)
    spec = json.loads(spec_found.group(1)) if spec_found else {}
    if state is None and STATE.search(body):
        state = json.loads(STATE.search(body).group(1))
    meta = json.loads(found.group(1)) if found else derive_meta(card_name((item.get("content") or {}).get("title")), body, spec, state)
    changes = spec.get("changes", {})
    if changes and not spec.get("base") and not meta.get("manual"):
        before = (state or {}).get("before") or {}
        meta["was"] = ", ".join(fmt(before.get(k, config_value(k))) for k in changes)
        meta["became"] = ", ".join(fmt(v) for v in changes.values())
    return meta


def boxed(value: str) -> str:
    """Значение в квадратных скобках (задача автора #62: «⬇️ ❌ [0.1] → [0.5] Тут описание»); список уже в скобках."""
    value = str(value or "—")
    return value if value.startswith("[") and value.endswith("]") else f"[{value}]"


def title_core(meta: dict) -> str:
    if meta.get("was", "—") in ("—", "") and meta.get("became", "—") in ("—", ""):
        return meta["short"]
    return f"{boxed(meta['was'])} → {boxed(meta['became'])} {meta['short']}"


def compose(meta: dict, spec_text: str | None, state: dict | None, results: list[str], maps: str, tree: str) -> str:
    """Тело карточки (задача автора #5): над картами — что пробуем коротко, подробно, было, стало; затем итоги и карты;
    служебные блоки прогонщика — свёрнуто; в конце — дерево прогонов."""
    changed = meta.get("was", "—") not in ("—", "") or meta.get("became", "—") not in ("—", "")
    parts = [f"**Что пробуем:** {meta['short']}", f"**Подробно:** {meta['full']}",
             f"**Было:** {boxed(meta['was']) if changed else '—'}", f"**Стало:** {boxed(meta['became']) if changed else '—'}"]
    if results:
        parts.append("\n".join(results).strip())
    if maps.strip():
        parts.append(maps.strip())
    service = ["<details><summary>Служебное</summary>", "", "```карточка", json.dumps(meta, ensure_ascii=False), "```"]
    if spec_text:
        service += ["", "```прогон", spec_text, "```"]
    if state is not None:
        service += ["", "```состояние", json.dumps(state, ensure_ascii=False), "```"]
    parts.append("\n".join(service + ["", "</details>"]))
    return "\n\n".join(parts) + (tree if tree else "")


def set_body(item: dict, text: str) -> None:
    content = item["content"]
    if text != content.get("body"):
        save_card(content, body=text)
        content["body"] = text


def migrate_card(item: dict, rows: dict) -> None:
    """Старую карточку — в новый вид (шапка, итоги, карты, свёрнутые служебные блоки, дерево)."""
    body = item["content"].get("body") or ""
    if CARD.search(body):
        return
    found = STATE.search(body)
    if found:
        write_body(item, json.loads(found.group(1)), rows)
        return
    spec_found, tree, maps = SPEC.search(body), TREE_BLOCK.search(body), MAPS.search(body)
    meta = card_meta(item)
    set_body(item, compose(meta, spec_found.group(1) if spec_found else None, None, [], maps.group(0) if maps else "",
                           tree.group(0) if tree else ""))


def tree_text(tree: dict, card: str | None, rows: dict | None = None, links: dict | None = None, names: dict | None = None) -> str:
    """Дерево прогонов плана, как дерево папок (автор 2026-11-21): узел — v3.720–723, тупик — v3.726–729.❌;
    ✅ — принят в опору, ⏳ — идёт; ◀ — узлы этой карточки (card — id карточки); 🏆 — самый лучший вариант на сейчас:
    текущая опора. links — id карточки → адрес: текст узла — ссылка на её тикет; names — id → «было ---> стало, что»."""
    import html
    children: dict = {}
    for node in tree["nodes"]:
        children.setdefault(node.get("parent"), []).append(node)
    for nodes in children.values():
        nodes.sort(key=lambda n: int(n["key"]))

    def line(node: dict) -> str:
        here = "  ◀" if card and (node.get("item") == card or card in node.get("items", [])) else ""
        cup = " 🏆" if node["key"] == tree.get("base") else ""
        text = node["name"] if node.get("name_fixed") else (names or {}).get(node.get("item")) or node.get("name") or node["card"]
        url = (links or {}).get(node.get("item"))
        if links is not None:
            text = f'<a href="{html.escape(url)}">{html.escape(text)}</a>' if url else html.escape(text)
        return f"{tag(tree['plan'], node['runs'])}{MARKS.get(node['status'], '')} {text}{cup}{here}"

    lines = []

    def walk(node: dict, indent: str, last: bool, root: bool) -> None:
        lines.append(line(node) if root else indent + ("└── " if last else "├── ") + line(node))
        below = children.get(node["key"], [])
        for i, child in enumerate(below):
            walk(child, "" if root else indent + ("    " if last else "│   "), i == len(below) - 1, False)

    for root in children.get(None, []):
        walk(root, "", True, True)
    return "\n".join(lines)


def tree_sync() -> None:
    """Дерево по доске: узел — карточка с прогонами (ключ — первый номер, item — id карточки); новый узел — ребёнок
    узла-родителя (авто-узлы, parent_id) или текущей опоры; принятый — новая опора; снятый паузой — убрать. Затем дерево —
    в конец текста каждой карточки."""
    tree = json.load(open(TREE_PATH, encoding="utf-8"))
    nodes = {n["key"]: n for n in tree["nodes"]}
    project = B.board()
    rows = log_rows()
    status = B.status_field(project)["name"]
    live = set()
    for item in project["items"]["nodes"]:
        content = item.get("content") or {}
        body = content.get("body") or ""
        found, spec_found = STATE.search(body), SPEC.search(body)
        if not found:
            continue
        state = json.loads(found.group(1))
        runs = sorted(int(n) for n in state.get("runs", {}).values())
        if not runs:
            continue
        spec = json.loads(spec_found.group(1)) if spec_found else {}
        key, column = str(runs[0]), item_fields(item).get(status)
        live.add(key)
        mark_ = "base" if state.get("kind") == "опора" else {GOOD: "good", BAD: "bad"}.get(column, "run")
        node = nodes.get(key)
        if node is None:
            parent = tree["base"]
            if spec.get("gen") and (spec.get("parent_id") or spec.get("parent")):
                parent = next((n["key"] for n in reversed(tree["nodes"]) if spec.get("parent_id") and n.get("item") == spec["parent_id"]),
                              next((n["key"] for n in reversed(tree["nodes"]) if n["card"] == card_name(spec.get("parent"))), parent))
            node = {"key": key, "runs": runs, "card": card_name(content["title"]), "item": item["id"], "parent": parent, "status": mark_}
            tree["nodes"].append(node)
            nodes[key] = node
        node.setdefault("item", item["id"])
        node["runs"] = sorted(set(node["runs"]) | set(runs))
        if mark_ == "good" and node["status"] != "good":
            tree["base"] = key
        node["status"] = mark_
    # Карточки-зёрна (схема «зерно = карточка»): узел — правка целиком, ключ — её первый прогон, item — карточка с ним.
    groups: dict = {}
    for item in project["items"]["nodes"]:
        state = card_state(item)
        if state and "seed" in state and state.get("run"):
            groups.setdefault(state["variant"], []).append((item, state, item_fields(item).get(status)))
    for vid, cards in groups.items():
        cards.sort(key=lambda c: int(c[1]["run"]))
        first, runs = cards[0][0], sorted(int(c[1]["run"]) for c in cards)
        columns = {c[2] for c in cards}
        mark_ = "base" if cards[0][1].get("kind") == "опора" else "good" if GOOD in columns else "bad" if BAD in columns else "run"
        key = str(runs[0])
        live.add(key)
        node = nodes.get(key)
        if node is None:
            spec, _ = card_spec(first)
            parent = tree["base"]
            if (spec or {}).get("gen") and (spec or {}).get("parent_id"):
                parent = next((n["key"] for n in reversed(tree["nodes"]) if n.get("item") == spec["parent_id"]
                               or spec["parent_id"] in n.get("items", [])), parent)
            node = {"key": key, "runs": runs, "card": card_name(first["content"]["title"]), "item": first["id"], "parent": parent,
                    "status": mark_}
            tree["nodes"].append(node)
            nodes[key] = node
        node["runs"], node["items"], node["variant"] = runs, [c[0]["id"] for c in cards], vid
        if mark_ == "good" and node["status"] != "good":
            tree["base"] = key
        node["status"] = mark_
    tree["nodes"] = [n for n in tree["nodes"] if n["status"] != "run" or n["key"] in live]
    json.dump(tree, open(TREE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    found = {"url": project["url"], "items": {"nodes": B.items("id databaseId")}}
    # Ссылка внутри сайта (без https://github.com) — чтобы открывалась в той же вкладке (атрибут target GitHub вырезает).
    site = re.sub(r"^https://github\.com", "", found["url"])
    links = {i["id"]: f"{site}/views/1?pane=issue&itemId={i['databaseId']}" for i in found["items"]["nodes"]}
    names = {i["id"]: re.sub(r" · зерно \d+$", "", TITLE_MARKS.sub("", (i.get("content") or {}).get("title") or "")) for i in project["items"]["nodes"]}
    busy, written = pair_busy(), 0
    for item in project["items"]["nodes"]:
        content = item.get("content") or {}
        if content.get("__typename") not in CARD_TYPES:
            continue
        if not CARD.search(content.get("body") or ""):
            # Задача автора — не карточка прогона: дерево в её текст не писать, попавшее раньше — убрать.
            if TREE_BLOCK.search(content.get("body") or ""):
                save_card(content, body=TREE_BLOCK.sub("", content.get("body") or "").rstrip() or " ")
            continue
        body = TREE_BLOCK.sub("", content.get("body") or "").rstrip()
        text = body + "\n\n### Дерево прогонов\n<pre>\n" + tree_text(tree, item["id"], rows, links, names) + "\n</pre>"
        if text != content.get("body"):
            save_card(content, body=text)
            written += 1
            # Дерево — во всех карточках, после решения переписывается десяток минут запросов к GitHub; пара за это время
            # может кончиться (2026-11-22: 846/847 упали через 30 с, а заметили через 4 мин). Пара кончилась — остаток
            # карточек в следующий раз (их текст всё равно будет отличаться), сначала решение и запуск.
            if busy and written % 5 == 0 and not pair_busy():
                say(f"дерево: пара кончилась — переписано {written} карточек, остальные — в следующий раз")
                return


def set_text(project: dict, item: dict, name: str, value: str) -> None:
    field = next((f for f in project["fields"]["nodes"] if f and f.get("name") == name), None)
    if field:
        B.graphql("""mutation($p: ID!, $i: ID!, $f: ID!, $t: String!) { updateProjectV2ItemFieldValue(input: {projectId: $p,
            itemId: $i, fieldId: $f, value: {text: $t}}) { projectV2Item { id } } }""",
                  {"p": project["id"], "i": item["id"], "f": field["id"], "t": value})


def mark(project: dict, item: dict, state: dict) -> None:
    """Поля «Версия» (версия плана) и «Прогоны» (номера) — серым на плитке; заголовок — без обозначения."""
    content = item["content"]
    title = PREFIX.sub("", content["title"])
    if title != content["title"]:
        save_card(content, title=title)
        content["title"] = title
    sync_card_fields(project, item, state)


def card_runs(item: dict, state: dict | None = None) -> list[int]:
    state = state if state is not None else (card_state(item) or {})
    runs = [int(n) for n in (state.get("runs") or {}).values() if n]
    if state.get("run"):
        runs.append(int(state["run"]))
    return sorted(set(runs))


def card_fields_expected(item: dict, state: dict | None = None) -> dict:
    """Поля «Версия» и «Прогоны» карточки (автор 2026-11-22: «тег версии / тег прогона должен быть актуальным и
    указываться при создании и изменении карточки»): есть прогоны — версия по их номерам; нет — версия из плана карточки
    («plan» блока «прогон», v4.6 → v4), иначе текущая; прогонов нет — «—»."""
    runs = card_runs(item, state)
    if runs:
        versions = sorted({version_of_run(n) for n in runs}, key=lambda v: plan_starts().get(v, 0))
        return {"Версия": ", ".join(versions), "Прогоны": span(runs)}
    have = item_fields(item)
    if have.get("Версия") and ({GOOD, BAD} & set(have.values())):
        return {"Версия": have["Версия"], "Прогоны": have.get("Прогоны") or "—"}  # законченная без прогонов — как была
    spec, _ = card_spec(item)
    found = re.match(r"(v\d+)\.", str((spec or {}).get("plan") or ""))
    return {"Версия": found.group(1) if found else plan_version((item.get("content") or {}).get("title") or ""), "Прогоны": "—"}


def sync_card_fields(project: dict, item: dict, state: dict | None = None) -> None:
    """Поставить «Версию» и «Прогоны», если на доске другие (только изменившиеся поля — по запросу на поле)."""
    have = item_fields(item)
    for name, value in card_fields_expected(item, state).items():
        if have.get(name) != value:
            set_text(project, item, name, value)
            nodes = item.setdefault("fieldValues", {"nodes": []})["nodes"]
            nodes[:] = [v for v in nodes if not (v and v.get("field", {}).get("name") == name)]
            nodes.append({"text": value, "field": {"name": name}})


def historical_base(state: dict, rows: dict) -> dict:
    """Опора, против которой карточку проверяли: по зёрнам — первые прогоны предков её узла в дереве (родитель, дед…)."""
    runs = sorted(int(n) for n in state.get("runs", {}).values())
    if not runs:
        return {}
    tree = json.load(open(TREE_PATH, encoding="utf-8"))
    nodes = {n["key"]: n for n in tree["nodes"]}
    node, base = nodes.get(str(runs[0])), {}
    parent = node.get("parent") if node else None
    while parent in nodes:
        if nodes[parent]["status"] == "good":  # опора — только принятые предки (у «полшага» / «обратно» родитель — закрытый узел)
            for number in nodes[parent]["runs"]:
                seed = (rows.get(int(number), {}).get("changes") or {}).get("ENCODER_RANDOM_SEED", 0)
                base.setdefault(str(seed), int(number))
        parent = nodes[parent].get("parent")
    return base


def write_body(item: dict, state: dict, rows: dict, project: dict | None = None) -> None:
    if project:
        mark(project, item, state)
    content = item["content"]
    body = content.get("body") or ""
    spec_found, tree = SPEC.search(body), TREE_BLOCK.search(body)
    meta = card_meta(item, state)
    base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
    if state.get("verdict"):  # законченная карточка — опора того времени: прогоны предков узла в дереве
        base = historical_base(state, rows) or base
    lines = ["### Итоги прогонов (прогонщик доски)", "",
             "| Зерно | Прогон | М1u | Опора | Δ | Монопятен из 10 (опора) | Пятен (опора) | Точек верха (опора) | Честная | Читатель |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    deltas = []
    for seed, number in state.get("runs", {}).items():
        entry = rows.get(int(number))
        ref = rows.get(int(base[str(seed)])) if str(seed) in base else None
        if not entry or m1u(entry) is None:
            lines.append(f"| {seed} | {number} | идёт | | | | | | | |")
            continue
        if ref is None or m1u(ref) is None:  # новое зерно — опоры на нём ещё нет
            lines.append(f"| {seed} | {number} | {m1u(entry):.3f} | — | — | {entry.get('whole_mean')} | {entry.get('top_m1_patches')} | "
                         f"{entry['points'][-1]} | {entry.get('top_honest', '')} | {round(100 * (entry.get('accuracy200') or 0))} % |")
            continue
        delta = m1u(entry) - m1u(ref)
        deltas.append(delta)
        lines.append(f"| {seed} | {number} | {m1u(entry):.3f} | {m1u(ref):.3f} | {delta:+.3f} | "
                     f"{entry.get('whole_mean')} ({ref.get('whole_mean')}) | {entry.get('top_m1_patches')} ({ref.get('top_m1_patches')}) | {entry['points'][-1]} ({ref['points'][-1]}) | "
                     f"{entry.get('top_honest', '')} | {round(100 * (entry.get('accuracy200') or 0))} % |")
    if deltas:
        lines.append(f"| Итого | | | | **{sum(deltas) / len(deltas):+.3f}** (в плюсе {sum(d > 0 for d in deltas)} из {len(deltas)}) | | | | | |")
    if state.get("verdict"):
        lines += ["", f"**Итог:** {state['verdict']}"]
    maps = "\n".join(maps_block(state.get("runs", {}), rows))
    set_body(item, compose(meta, spec_found.group(1) if spec_found else None, state, lines, maps, tree.group(0) if tree else ""))


def patches_better(state: dict, rows: dict, base: dict, mean: float) -> tuple[bool, str]:
    """Правило пятен (автор 2026-11-21: «научить правило видеть пятна»): М1u не хуже (Δ ≥ −0.01), пятен на верху меньше
    на КАЖДОМ зерне и точек верха столько же (±10 %) — классы собраны крупнее, а не просто точек стало меньше."""
    pairs = [(rows[int(n)], rows[int(base[str(s)])]) for s, n in state["runs"].items()]
    fewer = sum(new["top_m1_patches"] < ref["top_m1_patches"] for new, ref in pairs)
    same = all(abs(new["points"][-1] / ref["points"][-1] - 1) <= 0.10 for new, ref in pairs)
    change = sum(new["top_m1_patches"] - ref["top_m1_patches"] for new, ref in pairs) / len(pairs)
    text = f"пятен {change:+.1f} (меньше на {fewer} из {len(pairs)}, точек {'столько же' if same else 'не столько же'})"
    return mean >= -0.01 and fewer == len(pairs) and same, text


RULE = "v3.4"
WEAK = (2, 4, 5)  # слабые цифры опоры (М1 0.38 / 0.47 / 0.43, docs/DIRECTION.md)
BASE_SEEDS = [1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11]  # опора — на 12 зёрнах (разбор методики 2026-11-21)
FUTILITY_FROM = 4  # досрочное закрытие (v3.4, задача автора #119) — на этапах 4 и 5, перед решением по 12 зёрнам
FUTILITY_SIGMAS = 2.0  # насколько среднее оставшихся зёрен должно превзойти наблюдаемое, чтобы ещё был шанс
FUTILITY_SD_FLOOR = 0.05  # нижняя граница разброса Δ по зерну (опора: Δ М1u по зерну ±0.06)


def futility(state: dict, rows: dict, base: dict) -> str | None:
    """Правило v3.4 (разбор #119, 2026-11-22): на этапах 4–5 (8 и 10 зёрен) правка закрывается досрочно, если обоим путям
    решения по 12 зёрнам (М1u +0.025, слабые +0.045) нужно, чтобы оставшиеся зёрна дали в среднем больше, чем наблюдаемая
    Δ + FUTILITY_SIGMAS · sd / √остаток (Δ — к опоре на том же зерне: на 12 зёрнах это и есть сравнение со средним опоры).
    Пример: плотность 0.07 на 8 зёрнах Δ +0.002 при sd 0.065 — нужно +0.072 на 4 зёрнах, шанс ≈ 2 %; экономит 1–2 пары."""
    stage = state["stage"]
    if stage < FUTILITY_FROM or stage >= len(STAGES):
        return None
    pairs = [(rows[int(n)], rows[int(base[str(s)])]) for s, n in state["runs"].items()
             if str(s) in base and int(base[str(s)]) in rows and int(n) in rows]
    total = sum(len(p) for p in STAGES)
    left = total - len(pairs)
    if left <= 0 or len(pairs) < 2:
        return None
    m1_gate, weak_gate = CHECKPOINTS[len(STAGES)]
    parts = []
    for name, measure, gate in (("М1u", m1u, m1_gate), ("слабые 2, 4, 5", weak, weak_gate)):
        deltas = [measure(a) - measure(b) for a, b in pairs]
        mean = sum(deltas) / len(deltas)
        sd = max(FUTILITY_SD_FLOOR, (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5)
        need = (total * gate - sum(deltas)) / left
        reach = FUTILITY_SIGMAS * sd / left ** 0.5
        if need - mean <= reach:
            return None
        parts.append(f"{name}: Δ {mean:+.3f} по {len(deltas)} зёрнам, оставшимся {left} нужно {need:+.3f} в среднем "
                     f"(на {need - mean:.3f} выше, допуск {FUTILITY_SIGMAS:g}·{sd:.3f}/√{left} = {reach:.3f})")
    return f"этап {stage} ({len(pairs)} зёрен): досрочно закрыто — " + "; ".join(parts) + f" [правило {RULE}]"


def weak(entry: dict) -> float:
    values = entry.get("m1_digits_mean") or [0.0] * 10
    return sum(values[d] for d in WEAK) / len(WEAK)


def base_means(rows: dict, base: dict) -> tuple[float, float, int]:
    """Средние опоры по всем её зёрнам: М1u, М1 слабых цифр, сколько зёрен (разбор методики: ошибка опоры на одном зерне
    застывает и одинаково сдвигает все карточки — сравнивать со средним опоры, а не с её прогоном на том же зерне)."""
    found = [rows[int(n)] for n in base.values() if int(n) in rows and m1u(rows[int(n)]) is not None]
    return sum(m1u(e) for e in found) / len(found), sum(weak(e) for e in found) / len(found), len(found)


def decide(state: dict, rows: dict) -> tuple[str, str | None]:
    """Правило v3.3 (план v4, задача автора #88; анализ шума опоры по 12 зёрнам, sd М1u 0.033): решение только на 2, 6 и 12
    зёрнах (этапы 1, 3, 6 из пар STAGES; этапы между ними — дальше без решения, кроме досрочного закрытия v3.4 — futility). Этап 1 — Δ к опоре на ТЕХ ЖЕ зёрнах
    (среднее 12 зёрен выше на 0.013 — сдвиг подмножества сидел у порога 0), этапы 3 и 6 — к среднему опоры по 12 зёрнам.
    Пороги М1u −0.01 / +0.015 / +0.025; путь «слабые 2, 4, 5» +0.04 / +0.05 / +0.045 при М1u не хуже −0.01 и честной не ниже
    опоры −0.05; пятна и монопятна на каждом зерне — как раньше. Принять — только по 12 зёрнам: прогоны правки сразу
    становятся новой опорой. По модели: истинный +0.03 принимается в 67 % (v3.2 — 38 %), +0.02 — в 30 % (17 %), ложных 1 %."""
    base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
    stage = state["stage"]
    missing = [n for n in state["runs"].values() if int(n) not in rows]
    if missing:
        # Прогон без итога в журнале стенда (упавший, дубль карточки) — не основание ронять шаг (2026-11-22: KeyError(847)).
        say(f"решение без прогонов {sorted(int(n) for n in missing)}: их итога нет в журнале стенда")
        state["runs"] = {s: n for s, n in state["runs"].items() if int(n) in rows}
    # Настройка не действует (2026-11-22: v4.9 торможение 1.0 → 1.5 дало на зёрнах 1/3 тот же результат до бита, что
    # опора): все прогоны правки совпали с опорой на тех же зёрнах по М1u цифр, точкам и сходимости — закрыть сразу.
    paired = [(rows[int(n)], rows[int(base[str(s)])]) for s, n in state["runs"].items()
              if str(s) in base and int(base[str(s)]) in rows and int(n) in rows]
    if paired and all(a.get(key) == b.get(key) for a, b in paired for key in ("m1_digits_mean", "points", "honest")):
        return "bad", (f"этап {stage} ({len(paired)} зёрен): настройка не действует — все прогоны совпали с опорой на тех же "
                       f"зёрнах до бита (М1u по цифрам, точки карт, сходимость) [правило {RULE}]")
    if stage not in CHECKPOINTS:
        hopeless = futility(state, rows, base)
        return ("bad", hopeless) if hopeless else ("next", None)
    runs = [rows[int(n)] for n in state["runs"].values()]
    if stage == 1:
        same = [rows[int(base[str(s)])] for s in state["runs"] if str(s) in base and int(base[str(s)]) in rows]
        mean_base = sum(m1u(e) for e in same) / len(same)
        weak_base = sum(weak(e) for e in same) / len(same)
        honest_base = sum(float(e.get("top_honest") or 0) for e in same) / len(same)
        against = f"к опоре на тех же {len(same)} зёрнах"
    else:
        mean_base, weak_base, seeds = base_means(rows, base)
        found = [rows[int(n)] for n in base.values() if int(n) in rows]
        honest_base = sum(float(e.get("top_honest") or 0) for e in found) / len(found)
        against = f"к среднему опоры по {seeds} зёрнам"
    mean = sum(m1u(e) for e in runs) / len(runs) - mean_base
    weak_mean = sum(weak(e) for e in runs) / len(runs) - weak_base
    honest = sum(float(e.get("top_honest") or 0) for e in runs) / len(runs) - honest_base
    spots, spots_text = patches_better(state, rows, base, mean)
    pairs = [(rows[int(n)].get("whole_mean", 0), rows[int(base[str(s)])].get("whole_mean", 0)) for s, n in state["runs"].items()
             if str(s) in base]
    more = sum(a > b for a, b in pairs)
    spots = spots or (mean >= -0.01 and pairs and more == len(pairs))
    m1_gate, weak_gate = CHECKPOINTS[stage]
    by_m1 = mean >= m1_gate
    by_weak = weak_mean >= weak_gate and mean >= -0.01 and honest >= -0.05
    text = (f"Δ М1u {mean:+.3f} {against}; слабые 2, 4, 5 {weak_mean:+.3f}; честная {honest:+.3f}; {spots_text}; "
            f"монопятен больше на {more} из {len(pairs)} [правило {RULE}]")
    if by_m1 or by_weak or spots:
        why = "по М1u" if by_m1 else ("по слабым цифрам" if by_weak else "по пятнам")
        if stage < len(STAGES):
            return "next", None
        return "good", f"по {len(runs)} зёрнам {why}: {text} → принято, в опору; от узла — продолжение"
    return "bad", f"этап {stage} ({len(runs)} зёрен): {text} — ниже порога → закрыто"


def tuning_row(title: str, state: dict, verdict: str) -> None:
    path = os.path.join(ROOT, "docs", "TUNING.md")
    runs = tag(plan_version(title), state["runs"].values())
    text = open(path, encoding="utf-8").read().rstrip("\n")
    text += f"\n| {card_name(title)} (прогонщик доски) | {runs} | | | | | | | | {verdict} |"
    open(path, "w", encoding="utf-8").write(text)


def card_state(item: dict) -> dict | None:
    found = STATE.search((item.get("content") or {}).get("body") or "")
    return json.loads(found.group(1)) if found else None


def card_spec(item: dict) -> tuple[dict, str] | tuple[None, None]:
    found = SPEC.search((item.get("content") or {}).get("body") or "")
    return (json.loads(found.group(1)), found.group(1)) if found else (None, None)


def variant_of(item: dict) -> str:
    """Правка, к которой относится карточка-зерно (автор 2026-11-21: «одна карточка — один прогон… зерно = своя карточка»)."""
    state, (spec, _) = card_state(item) or {}, card_spec(item)
    return state.get("variant") or (spec or {}).get("variant") or item["id"]


def siblings(items: list[dict], vid: str) -> list[tuple[dict, dict]]:
    """Карточки-зёрна правки vid по новой схеме: (карточка, состояние)."""
    found = []
    for item in items:
        state = card_state(item)
        if state and "seed" in state and state.get("variant") == vid:
            found.append((item, state))
    return found


def variant_runs(cards: list[tuple[dict, dict]]) -> dict:
    return {str(state["seed"]): int(state["run"]) for _, state in cards if state.get("status") == "done" and state.get("run")}


def maps_pair(base_run: int | None, run: int | None, rows: dict) -> str:
    """В теле — только карта «было» (опора на этом зерне). Карта «стало» — в комментарии-итоге прогона
    (автор 2026-11-21: «в теле карты „как стало“ не показывай»)."""
    lines = []
    for title, number in (("Было — опора", base_run),):
        if not number or int(number) not in rows:
            continue
        try:
            url = map_url(int(number))
        except Exception as error:  # noqa: BLE001  сеть — карта будет при следующей записи
            say(f"карта {number} не выложена: {error!r}")
            url = None
        if url:
            lines += [f"**{title}:** {tag(PLAN, [number])}", "", f"![карта прогона {number}]({url})", ""]
    return "\n".join(["### Карты прогонов", ""] + lines) if lines else ""


def write_seed_card(item: dict, state: dict, rows: dict, project: dict | None, summary: str = "") -> None:
    """Тело карточки-зерна: шапка, строка этого прогона против опоры на том же зерне, итог правки целиком, карты
    «было / стало», служебное, дерево."""
    if project:
        mark(project, item, {"runs": {str(state["seed"]): state["run"]}} if state.get("run") else {})
    content = item["content"]
    body = content.get("body") or ""
    spec_found, tree = SPEC.search(body), TREE_BLOCK.search(body)
    meta = card_meta(item, state)
    base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
    if state.get("verdict") and state.get("variant_runs"):
        base = historical_base({"runs": state["variant_runs"]}, rows) or base
    seed, run = str(state["seed"]), state.get("run")
    ref_run = base.get(seed)
    lines = [f"**Прогон:** {tag(PLAN, [run]) if run else 'ещё не запущен'}, зерно {seed}, этап {state['stage']}", "",
             "| Зерно | Прогон | М1u | Опора | Δ | Слабые 2, 4, 5 (опора) | Монопятен (опора) | Пятен (опора) | Точек верха (опора) | Честная | Читатель |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    entry, ref = rows.get(int(run)) if run else None, rows.get(int(ref_run)) if ref_run else None
    if not entry or m1u(entry) is None:
        lines.append(f"| {seed} | {run or '—'} | {'идёт' if state.get('status') == 'running' else 'ждёт'} | | | | | | | | |")
    elif ref is None or m1u(ref) is None:
        lines.append(f"| {seed} | {run} | {m1u(entry):.3f} | — | — | {weak(entry):.3f} | {entry.get('whole_mean')} | {entry.get('top_m1_patches')} | "
                     f"{entry['points'][-1]} | {entry.get('top_honest', '')} | {round(100 * (entry.get('accuracy200') or 0))} % |")
    else:
        lines.append(f"| {seed} | {run} | {m1u(entry):.3f} | {m1u(ref):.3f} | {m1u(entry) - m1u(ref):+.3f} | {weak(entry):.3f} ({weak(ref):.3f}) | "
                     f"{entry.get('whole_mean')} ({ref.get('whole_mean')}) | {entry.get('top_m1_patches')} ({ref.get('top_m1_patches')}) | "
                     f"{entry['points'][-1]} ({ref['points'][-1]}) | {entry.get('top_honest', '')} | {round(100 * (entry.get('accuracy200') or 0))} % |")
    if state.get("verdict"):
        lines += ["", f"**Итог правки целиком:** {state['verdict']}"]
    elif summary:
        lines += ["", f"**Правка целиком:** {summary}"]
    set_body(item, compose(meta, spec_found.group(1) if spec_found else None, state, lines,
                           maps_pair(ref_run, run, rows), tree.group(0) if tree else ""))


def new_card(project: dict, item: dict, spec: dict, spec_text: str, state: dict, column: str) -> dict:
    """Карточка-зерно той же правки: шапка — та же, блок «прогон» с номером правки, своё состояние."""
    meta = card_meta(item)
    spec = {**spec, "variant": state["variant"]}
    body = compose(meta, json.dumps(spec, ensure_ascii=False), state, [], "", "")
    created = B.graphql("""mutation($p: ID!, $t: String!, $b: String) {
        addProjectV2DraftIssue(input: {projectId: $p, title: $t, body: $b}) { projectItem { id } } }""",
                        # Приоритет — с создания, первым (автор 2026-11-23: «в карточке ВСЕГДА должен быть приоритет!!!»).
                        {"p": project["id"], "t": f"{PRIORITY_EMOJI[item_priority(item)]} {title_core(meta)} · зерно {state['seed']}",
                         "b": body})
    found = to_issue(created["addProjectV2DraftIssue"]["projectItem"]["id"])
    move(project, found, column, f"карточка зерна {state['seed']}, этап {state['stage']} правки «{card_name(item['content']['title'])}»")
    return found


def seed_finish(project: dict, items: list[dict], rows: dict) -> str | None:
    """Пары карточек-зёрен: дождаться, записать каждую, решить правку целиком (правило v3.2) и разнести карточки."""
    groups: dict = {}
    for item in items:
        state = card_state(item)
        if state and "seed" in state and state.get("status") == "running":
            groups.setdefault((state["variant"], state["stage"]), []).append((item, state))
    waiting = None
    for (vid, stage), cards in groups.items():
        runs = [int(state["run"]) for _, state in cards]
        if any(item_fields(item).get("Метка") == "пауза" for item, _ in cards):
            stop(runs)
            for item, state in cards:
                state.update({"status": "queued", "run": None})
                write_seed_card(item, state, rows, project)
                move(project, item, QUEUE, f"пауза: пара {span(runs)} остановлена, итоги стёрты; после снятия метки — заново")
            say(f"пауза: пара {runs} остановлена; после снятия метки — заново")
            return "пауза"
        if not finished(runs):
            waiting = waiting or f"ждёт пару {runs} — {cards[0][0]['content']['title']}"
            continue
        rows.update(log_rows())  # finished() читает свежий журнал, rows — снимок начала шага: итог мог дописаться между ними
        for item, state in cards:
            state["status"] = "done"
        if any(m1u(rows.get(int(state["run"]), {})) is None for _, state in cards):
            for item, state in cards:
                state["verdict"] = "ошибка прогона: в журнале стенда нет итога — разбирает Claude"
                write_seed_card(item, state, rows, project)
                tail = ""
                try:
                    tail = open(os.path.join(STAND, f"tune2-{state['run']}.err"), encoding="utf-8", errors="replace").read()[-1500:]
                except OSError:
                    pass
                move(project, item, ANALYSIS, f"ОШИБКА прогона {tag(PLAN, [state['run']])}: в журнале стенда нет итога."
                     + (f"\n\nХвост журнала ошибок:\n```\n{tail}\n```" if tail.strip() else ""))
            say(f"ошибка прогона {runs}")
            return "ошибка"
        everyone = siblings(items, vid)
        done_ids = {item["id"] for item, _ in cards}
        everyone = [(i, cards[[c["id"] for c, _ in cards].index(i["id"])][1]) if i["id"] in done_ids else (i, s) for i, s in everyone]
        vruns = variant_runs(everyone)
        first_item = min(everyone, key=lambda pair: int(pair[1].get("run") or 10 ** 9))[0]
        spec, spec_text = card_spec(first_item)
        agg = {"stage": stage, "runs": vruns, "before": cards[0][1].get("before", {})}
        if (spec or {}).get("check"):
            # Карточка-проверка (задача #15): одна пара, без принятия / отказа — числа для разбора Claude (сравнение
            # вариантов на одном и том же другом наборе картинок).
            verdict = ("проверка: М1u " + ", ".join(f"зерно {s}: {m1u(rows[n]):.3f}" for s, n in vruns.items())
                       + f"; слабые 2, 4, 5 " + ", ".join(f"{weak(rows[n]):.3f}" for n in vruns.values()) + " — разбирает Claude")
            for item, state in everyone:
                state.update({"verdict": verdict, "variant_runs": vruns})
                write_seed_card(item, state, rows, project)
                move(project, item, ANALYSIS, run_report(item, state, rows, "проверка без принятия и отказа — сравнение вариантов "
                                                         "на других картинках делает Claude; карточка ждёт в «Анализе»"))
            tuning_row(first_item["content"]["title"], agg, verdict)
            say(f"{verdict} — {first_item['content']['title']}")
            return "итог"
        if cards[0][1].get("kind") == "опора":
            path = os.path.join(HERE, "base_runs.json")
            base = json.load(open(path, encoding="utf-8"))
            verdict = "опора: М1u " + ", ".join(f"зерно {s}: {m1u(rows[n]):.3f}" for s, n in vruns.items()) + " → записана в base_runs.json"
            for item, state in everyone:
                state.update({"verdict": verdict, "variant_runs": vruns})
                write_seed_card(item, state, rows, project)
                move(project, item, GOOD, run_report(item, state, rows, "прогон записан в опору (base_runs.json) — с ним сравниваются правки"))
            base.update(vruns)
            json.dump(base, open(path, "w", encoding="utf-8"))
            tuning_row(first_item["content"]["title"], agg, verdict)
            say(f"{verdict} — {first_item['content']['title']}")
            return "итог"
        outcome, verdict = decide(agg, rows)
        base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
        mean_base = base_means(rows, base)[0]
        mean = sum(m1u(rows[n]) for n in vruns.values()) / len(vruns) - mean_base
        if outcome == "next":
            summary = f"этап {stage} пройден (Δ М1u {mean:+.3f} к среднему опоры по {len(vruns)} зёрнам) → этап {stage + 1} в очереди"
            for item, state in cards:
                write_seed_card(item, state, rows, project, summary)
                move(project, item, ANALYSIS, run_report(item, state, rows,
                     f"этап {stage} правки пройден (Δ М1u {mean:+.3f} к среднему опоры по {len(vruns)} зёрнам); зерно ждёт в «Анализе» "
                     f"остальные зёрна правки — решение по всем {sum(len(s) for s in STAGES)} зёрнам; этап {stage + 1} "
                     f"(зёрна {', '.join(map(str, STAGES[stage]))}) — в очереди первым"))
            seed = STAGES[stage][0]
            state = {"variant": vid, "seed": seed, "stage": stage + 1, "run": None, "status": "queued", "await": True,
                     "before": cards[0][1].get("before", {}), "base": cards[0][1].get("base", {})}
            new_card(project, first_item, spec, spec_text, state, QUEUE)
            say(f"этап {stage + 1} — в очередь — {first_item['content']['title']}")
            return "итог"
        done_now = {item["id"] for item, _ in cards}
        after = ("правка принята и записана в опору; от узла растёт продолжение" if outcome == "good"
                 else "правка закрыта; от опоры — «полшага» или «обратно», если эффект был заметен")
        for item, state in everyone:
            state.update({"verdict": verdict, "variant_runs": vruns})
            write_seed_card(item, state, rows, project)
            why = (run_report(item, state, rows, after) if item["id"] in done_now
                   else f"решение по правке целиком: {verdict}\n\n" + map_line(item, state.get("run"), "Карта этого зерна, как стало"))
            move(project, item, GOOD if outcome == "good" else BAD, why)
        tuning_row(first_item["content"]["title"], agg, verdict)
        say(f"{'ХОРОШО' if outcome == 'good' else 'плохо'}: {verdict} — {first_item['content']['title']}")
        grow(project, first_item, agg, outcome, mean)  # до accept: опора ещё прежняя
        if outcome == "good":
            accept(project, first_item, agg)
        return "итог"
    return waiting


def seed_launch(project: dict, items: list[dict], rows: dict) -> str | None:
    """Взять сверху Очереди карточку-зерно (или свежую правку), добавить к ней карточку второго зерна этапа, запустить пару."""
    status = B.status_field(project)["name"]

    def queued(item: dict) -> bool:
        content, fields = item.get("content") or {}, item_fields(item)
        if fields.get(status) != QUEUE or fields.get("Метка") or content.get("__typename") not in CARD_TYPES:
            return False
        spec, _ = card_spec(item)
        state = card_state(item)
        return bool(spec) and (state is None or ("seed" in state and state.get("status") == "queued"))

    # Порядок взятия (автор, #66: «задачи должны браться сначала по приоритету, а потом по времени»): начатые этапы,
    # потом приоритет, внутри — раньше созданные (показ в колонках — по времени, новые вверху).
    # Брать из Очереди всегда сверху (автор, комментарий к #66): порядок доски — приоритет, потом новые выше.
    candidates = [i for i in items if queued(i)]
    if not candidates:
        return None
    item = candidates[0]
    spec, spec_text = card_spec(item)
    state = card_state(item) or {}
    vid, stage = variant_of(item), state.get("stage", 1)
    seeds = spec.get("seeds") if spec.get("base") else STAGES[stage - 1]
    before = state.get("before") or {key: config_value(key) for key in spec.get("changes", {})}
    base_state = state.get("base") or base_now()
    cards = [(i, s) for i, s in siblings(items, vid) if s.get("stage") == stage and s.get("status") == "queued"]
    if not cards or item["id"] not in [i["id"] for i, _ in cards]:
        state = {"variant": vid, "seed": seeds[0], "stage": stage, "run": None, "status": "queued", "before": before, "base": base_state}
        if spec.get("base"):
            state["kind"] = "опора"
        cards = [(item, state)] + cards
    have = {int(s["seed"]) for _, s in cards}
    for seed in seeds:
        if seed not in have:
            extra = {"variant": vid, "seed": seed, "stage": stage, "run": None, "status": "queued", "before": before, "base": base_state}
            if spec.get("base"):
                extra["kind"] = "опора"
            cards.append((new_card(project, item, spec, spec_text, extra, QUEUE), extra))
    cards = sorted(cards, key=lambda pair: seeds.index(int(pair[1]["seed"])) if int(pair[1]["seed"]) in seeds else 99)[:len(seeds)]
    numbers = [next_number()]
    numbers.append(numbers[0] + 1)
    for (card, card_st), number in zip(cards, numbers):
        card_st.update({"run": number, "status": "running", "await": False})
    # План ускорения П1 (docs/SPEEDUP.md): пара стартует сразу, карточки пишутся уже во время её счёта.
    launch(numbers, [int(s["seed"]) for _, s in cards], spec.get("changes", {}), spec.get("label", item["content"]["title"]))
    for (card, card_st), number in zip(cards, numbers):
        write_seed_card(card, card_st, rows, project)
        move(project, card, RUN, f"запущен прогон {tag(PLAN, [number])}: зерно {card_st['seed']}, этап {card_st['stage']} — "
                                 f"{spec.get('label', '')}; ~8 мин")
    return f"взята — {card_name(item['content']['title'])} (зёрна {[int(s['seed']) for _, s in cards]}, прогоны {numbers})"


def housekeeping(project: dict) -> None:
    """Учёт доски, не нужный для решения и запуска пары: комментарии, план, версия плана, карты, порядок и поля."""
    try:
        check_comments(project)
    except Exception as error:  # noqa: BLE001
        say(f"комментарии не проверены: {error!r}")
    try:
        plan_sync(project)  # новые строки таблицы плана — карточки в Очередь (задача автора #88)
    except Exception as error:  # noqa: BLE001
        say(f"план не прочитан: {error!r}")
    try:
        ensure_plan_start()  # новая версия плана в plan.md — с какого прогона
    except Exception as error:  # noqa: BLE001
        say(f"версия плана не записана: {error!r}")
    try:
        maps_resend(project)  # карты, не выложившиеся при итоге (автор 2026-11-22: карта в комментарии — обязательно)
    except Exception as error:  # noqa: BLE001
        say(f"карты не досланы: {error!r}")
    try:
        order_queue(project)  # пришедшие карточки — наверх, поля «Версия» и «Прогоны»
    except Exception as error:  # noqa: BLE001
        say(f"порядок очереди не обновлён: {error!r}")


def step() -> str:
    """Шаг прогонщика. План ускорения П1 (docs/SPEEDUP.md, 2026-11-22): сначала решение по законченной паре и сразу запуск
    следующей (доска перечитывается один раз — чтобы увидеть карточку следующего этапа), учёт доски — после запуска."""
    project = B.board()
    rows = log_rows()
    items = project["items"]["nodes"]
    try:
        status = pair_step(project, items, rows)
        if status == "итог" and not pair_busy():
            project = B.board()
            relaunched = launch_step(project, project["items"]["nodes"], log_rows())
            status = f"итог; {relaunched}" if relaunched else status
    finally:
        housekeeping(project)
    return status


def pair_step(project: dict, items: list[dict], rows: dict) -> str:
    """Законченная пара — решение; иначе — запуск следующей карточки, если стенд свободен."""
    # 1. Карточка в работе (есть «состояние» с незаконченной парой) — дождаться пары и решить.
    waiting = ""
    for item in items:
        content = item.get("content") or {}
        found = STATE.search(content.get("body") or "")
        if not found:
            continue
        state = json.loads(found.group(1))
        if state.get("verdict") or not state.get("pending"):
            continue
        fields = item_fields(item)
        if fields.get("Метка") == "пауза":
            if not finished(state["pending"]):
                stop(state["pending"])
            clear(item)
            say(f"пауза: прогон остановлен, итоги стёрты; после снятия метки — заново — {content['title']}")
            return "пауза"
        if not finished(state["pending"]):
            waiting = waiting or f"ждёт пару {state['pending']} — {content['title']}"
            continue  # сначала разобрать законченные пары других карточек
        state["pending"] = []
        if any(m1u(rows.get(int(n), {})) is None for n in state["runs"].values()):
            state["verdict"] = "ошибка прогона: в журнале стенда нет итога — разбирает Claude"
            write_body(item, state, rows, project)
            move(project, item, ANALYSIS)
            say(state["verdict"] + f" — {content['title']}")
            return "ошибка"
        move(project, item, ANALYSIS)
        if state.get("kind") == "опора":
            # Карточка-опора: прогоны текущей опоры (STAND/base.json) на своих зёрнах — записать их в base_runs.json.
            path = os.path.join(HERE, "base_runs.json")
            base = json.load(open(path, encoding="utf-8"))
            values = ", ".join(f"зерно {s}: {m1u(rows[int(n)]):.3f} (было {m1u(rows[int(base[s])]):.3f})" if s in base
                               else f"зерно {s}: {m1u(rows[int(n)]):.3f}" for s, n in state["runs"].items())
            state["verdict"] = f"опора: М1u {values} → записана в base_runs.json"
            write_body(item, state, rows, project)
            base.update({s: int(n) for s, n in state["runs"].items()})
            json.dump(base, open(path, "w", encoding="utf-8"))
            move(project, item, GOOD)
            tuning_row(content["title"], state, state["verdict"])
            say(f"опора: {state['verdict']} — {content['title']}")
            return "итог"
        outcome, verdict = decide(state, rows)
        if outcome == "next":
            if pair_busy():  # на стенде идёт другая пара — следующий этап после неё (не больше одной пары)
                return f"ждёт стенд для этапа {state['stage'] + 1} — {content['title']}"
            # Флоу автора 2026-11-21: «Анализ — отправляет либо в плохие, либо в хорошие, либо обновляет и в очередь, либо в
            # тодо»: следующий этап — карточка обратно в Очередь (начатые этапы берутся первыми).
            state["stage"] += 1
            state["await"] = True
            write_body(item, state, rows, project)
            move(project, item, QUEUE)
            say(f"этап {state['stage']} — в очередь — {content['title']}")
            return "итог"
        state["verdict"] = verdict
        write_body(item, state, rows, project)
        move(project, item, GOOD if outcome == "good" else BAD)
        tuning_row(content["title"], state, verdict)
        say(f"{'ХОРОШО' if outcome == 'good' else 'плохо'}: {verdict} — {content['title']}")
        base = json.load(open(os.path.join(HERE, "base_runs.json"), encoding="utf-8"))
        mean_base = base_means(rows, base)[0]
        grow(project, item, state, outcome, sum(m1u(rows[int(n)]) for n in state["runs"].values()) / len(state["runs"]) - mean_base)  # до accept: опора ещё прежняя
        if outcome == "good":
            accept(project, item, state)
        return "итог"
    found = seed_finish(project, items, rows)
    if found in ("итог", "пауза", "ошибка"):
        return found
    waiting = waiting or found
    if waiting:
        return waiting
    return launch_step(project, items, rows) or "очередь пуста"


def launch_step(project: dict, items: list[dict], rows: dict) -> str | None:
    """2. Новая карточка: первая сверху в очереди, без метки, с блоком «прогон»; None — запускать нечего."""
    if pair_busy():
        return "стенд занят чужой парой"
    status = B.status_field(project)["name"]
    def waiting_stage(item: dict) -> dict | None:
        found = STATE.search((item.get("content") or {}).get("body") or "")
        state = json.loads(found.group(1)) if found else None
        legacy = state and "runs" in state and "seed" not in state  # старая схема: карточка = правка
        return state if legacy and state.get("await") and not state.get("verdict") else None

    # строго по доске сверху вниз (автор 2026-11-22: «разбираться колонка тоже должна сверху»); следующий этап правки
    # встаёт наверх Очереди сам (move → to_top), поэтому и берётся первым
    for item in items:
        content = item.get("content") or {}
        fields = item_fields(item)
        if fields.get(status) not in (QUEUE, RUN) or fields.get("Метка") or content.get("__typename") not in CARD_TYPES:
            continue
        spec_found = SPEC.search(content.get("body") or "")
        state = waiting_stage(item)
        if spec_found and state:  # следующий этап карточки, вернувшейся из Анализа
            seeds = STAGES[state["stage"] - 1]
            numbers = [next_number()]
            numbers.append(numbers[0] + 1)
            spec = json.loads(spec_found.group(1))
            state["runs"].update({str(s): n for s, n in zip(seeds, numbers)})
            state["pending"], state["await"] = numbers, False
            launch(numbers, seeds, spec.get("changes", {}), spec.get("label", content["title"]))  # П1: сначала запуск
            write_body(item, state, rows, project)
            move(project, item, RUN)
            return f"этап {state['stage']} — {content['title']}"
        continue  # свежие правки — по новой схеме «зерно = карточка» (seed_launch)
    return seed_launch(project, items, rows)


def main() -> None:
    once = "--once" in sys.argv
    say("прогонщик запущен")
    last = None
    first = True  # дерево — после первого шага, не до него: сначала решение по законченной паре и запуск (2026-11-22)
    while True:
        started = time.time()
        try:
            status = step()
        except Exception as error:  # noqa: BLE001  сеть / GitHub — повторить позже
            status = f"ошибка шага: {error!r}"
        changed = status != last
        if changed:
            say(status)
            last = status
        if status == "итог":
            # Решение без запуска (Очередь была пуста, а учёт доски мог завести карточки) — сразу шаг запуска; дерево —
            # после запуска или при пустой Очереди (2026-11-22: стенд стоял 5 мин, пока переписывалось дерево).
            time.sleep(3)
            continue
        if first or status.startswith(("взята", "этап", "итог", "пауза", "ошибка", "прогонщик")) or (status == "очередь пуста" and changed):
            first = False
            try:
                tree_sync()
            except Exception as error:  # noqa: BLE001
                say(f"дерево не обновлено: {error!r}")
        if once:
            return
        if status in ("итог", "ошибка"):
            time.sleep(3)
            continue
        # Пара кончилась, пока шёл шаг или дерево (2026-11-22: 854/855 ждали решения полторы минуты) — сразу новый шаг.
        done = os.path.join(STAND, "tune_pair.done")
        if os.path.exists(done) and os.path.getmtime(done) > started:
            time.sleep(3)
            continue
        # П1: пока идёт пара — локальная проверка её конца раз в 5 с; полный шаг (доска, комментарии) — раз в минуту.
        waited = 0
        busy = pair_busy()
        while waited < 60:
            time.sleep(5)
            waited += 5
            if busy and not pair_busy():
                break


if __name__ == "__main__":
    main()
