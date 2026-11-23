# -*- coding: utf-8 -*-
"""Доска прогонов на GitHub Projects: прочитать флоу (описания доски и колонок) и аккуратно залить план карточками.

    python tools/tuning/board.py read                                   — доска: описание, колонки с описаниями, карточки
    python tools/tuning/board.py plan tools/tuning/board_v3.json MAP    — что будет залито (ничего не пишет)
    python tools/tuning/board.py plan tools/tuning/board_v3.json MAP --apply — залить: новые карточки-черновики и колонки
    python tools/tuning/board.py card "заголовок" "колонка" [текст.md]  — одна карточка: создать / переложить / обновить текст

MAP — JSON «статус плана → название колонки доски», например {"accepted": "Done", "closed": "Done", "todo": "Todo"}; без MAP
карточки ложатся без колонки. Карточка с тем же заголовком на доске уже есть — не дублируется, только сверяется колонка.
Токен и адрес доски — из .env (GITHUB_TOKEN, GITHUB_URL = https://github.com/users/<логин>/projects/<номер>/…). Нужен
классический токен с областью project: токен с точными правами (github_pat_…) доску личного аккаунта не видит.
Автор 2026-11-21: «я дал тебе канбан для прогонов, чтобы я в нём мог трекать прогресс… аккуратно загрузи туда v3 план».
"""
import http.client
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def env() -> dict:
    values = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


CONF = env()
TOKEN = CONF.get("GITHUB_TOKEN", "")
MATCH = re.search(r"/(users|orgs)/([^/]+)/projects/(\d+)", CONF.get("GITHUB_URL", ""))
if not MATCH:
    sys.exit("GITHUB_URL в .env должен вести на доску: https://github.com/users/<логин>/projects/<номер>")
KIND, OWNER, NUMBER = ("user" if MATCH.group(1) == "users" else "organization"), MATCH.group(2), int(MATCH.group(3))


# Автор 2026-11-21: «у моего токена не должно быть доступа к коду. только в доске». Доска — GITHUB_TOKEN (классический,
# только область project); задачи репозитория (комментарии, заголовки, тексты, создание, метки) — GITHUB_ISSUES_TOKEN
# (токен с точными правами: только DAMP, Issues — чтение и запись, Contents — нет доступа). Нет второго — пока прежний.
ISSUES_TOKEN = CONF.get("GITHUB_ISSUES_TOKEN") or TOKEN
ISSUE_WRITES = re.compile(r"\b(addComment|updateIssue|closeIssue|reopenIssue|createIssue|addLabelsToLabelable|"
                          r"updateIssueComment|deleteIssueComment)\b")


def token_for(query: str) -> str:
    return ISSUES_TOKEN if ISSUE_WRITES.search(query) else TOKEN


def graphql(query: str, variables: dict | None = None) -> dict:
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Authorization": f"bearer {token_for(query)}", "Content-Type": "application/json"},
    )
    # Тайм-аут и повторы: 2026-11-21 прогонщик повис навсегда на оборванном ответе GitHub (SSL read без тайм-аута).
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # ответ обычно 1–5 с; 30 — чтобы подвисший запрос не держал минуту
                found = json.load(response)
            break
        except (TimeoutError, OSError, ValueError, http.client.HTTPException):  # IncompleteRead — оборванный ответ
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    if found.get("errors"):
        messages = "; ".join(error.get("message", "") for error in found["errors"])
        hint = " — нужен классический токен с областью project" if "personal access token" in messages else ""
        # Исключение, а не выход: разовая ошибка GitHub («Something went wrong…», 2026-11-23) роняла весь прогонщик —
        # SystemExit не ловится его `except Exception`; теперь шаг ловит и повторяет.
        raise RuntimeError(f"GitHub: {messages}{hint}")
    return found["data"]


ITEM_FIELDS = """id createdAt content { __typename ... on DraftIssue { id title body } ... on Issue { id title body number state } }
        fieldValues(first: 20) { nodes { ... on ProjectV2ItemFieldSingleSelectValue { name updatedAt field { ... on ProjectV2SingleSelectField { name } } }
                                          ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2Field { name } } } } }"""


def items(fragment: str) -> list[dict]:
    """Все карточки доски постранично (по 40: ответ по 100 карточек ~1 МБ, GitHub его обрывает — IncompleteRead): fragment — поля карточки в запросе GraphQL."""
    found, cursor = [], None
    while True:
        page = graphql("""query($owner: String!, $number: Int!, $after: String) { %s(login: $owner) { projectV2(number: $number) {
          items(first: 40, after: $after) { pageInfo { hasNextPage endCursor } nodes { %s } } } } }""" % (KIND, fragment),
                       {"owner": OWNER, "number": NUMBER, "after": cursor})[KIND]["projectV2"]["items"]
        found += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return found
        cursor = page["pageInfo"]["endCursor"]


def board() -> dict:
    query = """query($owner: String!, $number: Int!) { %s(login: $owner) { projectV2(number: $number) {
      id title shortDescription readme url
      fields(first: 50) { nodes { __typename ... on ProjectV2SingleSelectField { id name options { id name description } }
                                  ... on ProjectV2Field { id name dataType } } }
    } } }""" % KIND
    project = graphql(query, {"owner": OWNER, "number": NUMBER})[KIND]["projectV2"]
    project["items"] = {"nodes": items(ITEM_FIELDS)}
    return project


def status_field(project: dict) -> dict | None:
    selects = [f for f in project["fields"]["nodes"] if f.get("__typename") == "ProjectV2SingleSelectField"]
    return next((f for f in selects if f["name"].lower() in ("status", "статус")), selects[0] if selects else None)


def item_status(item: dict, field: dict | None) -> str:
    for value in item["fieldValues"]["nodes"]:
        if value and value.get("field", {}).get("name") == (field or {}).get("name"):
            return value.get("name", "")
    return ""


def read() -> None:
    project = board()
    print(f"Доска: {project['title']}\nКратко: {project.get('shortDescription') or '—'}\n\nОписание (README):\n{project.get('readme') or '—'}\n")
    field = status_field(project)
    for f in project["fields"]["nodes"]:
        if f.get("__typename") == "ProjectV2SingleSelectField":
            print(f"Поле «{f['name']}»{' (колонки)' if f is field else ''}:")
            for option in f["options"]:
                print(f"  • {option['name']}: {option.get('description') or '—'}")
    print(f"\nКарточек: {len(project['items']['nodes'])}")
    for item in project["items"]["nodes"]:
        content = item.get("content") or {}
        print(f"  [{item_status(item, field) or '—'}] {content.get('title', '?')}")


def plan(path: str, mapping_path: str | None, apply: bool) -> None:
    cards = json.load(open(path, encoding="utf-8"))["cards"]
    mapping = json.load(open(mapping_path, encoding="utf-8")) if mapping_path else {}
    project = board()
    field = status_field(project)
    options = {option["name"]: option["id"] for option in (field or {}).get("options", [])}
    missing = sorted({column for column in mapping.values() if column not in options})
    if missing:
        sys.exit(f"на доске нет колонок {missing}; есть: {sorted(options)}")
    existing = {((item.get("content") or {}).get("title") or ""): item for item in project["items"]["nodes"]}
    print(f"{'ЗАЛИВКА' if apply else 'ПРОБА (ничего не пишется)'}: доска «{project['title']}», карточек плана {len(cards)}")
    for card in cards:
        column = mapping.get(card["status"])
        item = existing.get(card["title"])
        action = "есть" if item else "новая"
        now = item_status(item, field) if item else ""
        move = f" → {column}" if column and column != now else ""
        print(f"  {action:5} [{now or '—'}{move}] {card['title']}")
        if not apply:
            continue
        if item is None:
            created = graphql("""mutation($project: ID!, $title: String!, $body: String) {
                addProjectV2DraftIssue(input: {projectId: $project, title: $title, body: $body}) { projectItem { id } } }""",
                              {"project": project["id"], "title": card["title"], "body": card.get("body", "")})
            item_id = created["addProjectV2DraftIssue"]["projectItem"]["id"]
        else:
            item_id = item["id"]
            content = item.get("content") or {}
            if content.get("__typename") == "DraftIssue" and card.get("body") and card["body"] != content.get("body"):
                graphql("""mutation($draft: ID!, $body: String!) { updateProjectV2DraftIssue(input: {draftIssueId: $draft, body: $body}) {
                    draftIssue { id } } }""", {"draft": content["id"], "body": card["body"]})
        if column and column != now:
            graphql("""mutation($project: ID!, $item: ID!, $field: ID!, $option: String!) {
                updateProjectV2ItemFieldValue(input: {projectId: $project, itemId: $item, fieldId: $field,
                  value: {singleSelectOptionId: $option}}) { projectV2Item { id } } }""",
                    {"project": project["id"], "item": item_id, "field": field["id"], "option": options[column]})
    print("готово" if apply else "проба: для записи добавьте --apply")


def card(title: str, column: str, body_path: str | None) -> None:
    """Одна карточка: создать, если нет; текст — из файла (Markdown); положить в колонку column (флоу доски)."""
    import tempfile

    body = open(body_path, encoding="utf-8").read() if body_path else ""
    spec = {"cards": [{"title": title, "status": "x", "body": body}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as one,             tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as where:
        json.dump(spec, one, ensure_ascii=False)
        json.dump({"x": column}, where, ensure_ascii=False)
    plan(one.name, where.name, True)
    os.remove(one.name)
    os.remove(where.name)


def scopes() -> None:
    """Доступ к доске и области токенов (автор 2026-11-21: «у моего токена не должно быть доступа к коду. только в доске»)."""
    project = graphql("""query($o: String!, $n: Int!) { %s(login: $o) { projectV2(number: $n) { title } } }""" % KIND,
                      {"o": OWNER, "n": NUMBER})[KIND]["projectV2"]
    print(f"доска: «{project['title']}» — доступна")
    for name, token in (("GITHUB_TOKEN", TOKEN), ("GITHUB_ISSUES_TOKEN", CONF.get("GITHUB_ISSUES_TOKEN"))):
        if not token:
            print(f"{name}: нет")
            continue
        request = urllib.request.Request("https://api.github.com/user", headers={"Authorization": f"bearer {token}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            found = response.headers.get("X-OAuth-Scopes")
        kind = "классический" if token.startswith("ghp_") else "с точными правами" if token.startswith("github_pat_") else "другой"
        warn = " — ВНИМАНИЕ: есть доступ к коду (repo), нужен только project" if name == "GITHUB_TOKEN" and found and "repo" in found.split(", ") else ""
        print(f"{name}: {kind}, области: {found or '— (с точными правами области не показываются)'}{warn}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("read", "plan", "card", "scopes"):
        sys.exit(__doc__)
    if sys.argv[1] == "scopes":
        scopes()
    elif sys.argv[1] == "read":
        read()
    elif sys.argv[1] == "card":
        card(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        args = [a for a in sys.argv[2:] if a != "--apply"]
        plan(args[0], args[1] if len(args) > 1 else None, "--apply" in sys.argv)
