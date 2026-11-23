// The reader Mini App (plan 2026-11-20-kirjakaveri-reader, "П2"): the list
// of books and chapters, and one chapter as paragraphs of word spans.
// No build step. Book text is only ever set through textContent.
"use strict";

const tg = window.Telegram && window.Telegram.WebApp;
const app = document.getElementById("app");

// What a chapter that can't be read yet says instead (plan, "Старт П2").
const STATUS_TEXT = {
  queued: "переводится",
  translating: "переводится",
  failed: "не переведена",
  pending: "появится, когда дочитаешь предыдущую",
};
const OPEN_FROM_BOT = "Открой читалку кнопкой «📖 Читать» в боте.";

class ApiError extends Error {
  constructor(status) {
    super(`HTTP ${status}`);
    this.status = status;
  }
}

async function api(path) {
  const response = await fetch(path, {
    headers: { Authorization: "tma " + (tg ? tg.initData : "") },
  });
  if (!response.ok) throw new ApiError(response.status);
  return response.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "onclick") node.addEventListener("click", value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child != null) node.append(child);
  }
  return node;
}

function show(...nodes) {
  app.replaceChildren(...nodes);
  window.scrollTo(0, 0);
}

function showError(error) {
  const text =
    error instanceof ApiError && error.status === 401
      ? OPEN_FROM_BOT
      : "Не получилось загрузить. Попробуй ещё раз.";
  show(el("p", { class: "muted" }, text));
}

function chapterSubtitle(chapter) {
  return [chapter.part, chapter.title].filter(Boolean).join(" · ");
}

function setBack(handler) {
  if (!tg) return;
  tg.BackButton.offClick(setBack.current || (() => {}));
  setBack.current = handler;
  if (handler) {
    tg.BackButton.onClick(handler);
    tg.BackButton.show();
  } else {
    tg.BackButton.hide();
  }
}

// --- list of books ------------------------------------------------------------

async function showBooks() {
  setBack(null);
  let data;
  try {
    data = await api("api/books");
  } catch (error) {
    showError(error);
    return;
  }
  if (data.books.length === 0) {
    show(
      el("h1", {}, "Книг пока нет"),
      el("p", { class: "muted" }, "Пришли книгу боту файлом .txt - переведу её на финский."),
    );
    return;
  }
  const nodes = [];
  for (const book of data.books) {
    nodes.push(el("h2", {}, book.title));
    const list = el("ul", { class: "chapters" });
    for (const chapter of book.chapters) {
      const ready = chapter.status === "done";
      list.append(
        el(
          "li",
          ready ? { class: "ready", onclick: () => showChapter(chapter.id) } : {},
          el("span", {}, `Глава ${chapter.number}`, el("br"),
            el("span", { class: "muted small" }, chapterSubtitle(chapter))),
          el("span", { class: "status" }, ready ? "" : STATUS_TEXT[chapter.status] || ""),
        ),
      );
    }
    nodes.push(list);
  }
  show(...nodes);
}

// --- one chapter --------------------------------------------------------------

function paragraph(pieces) {
  // [gap, word, gap, word, ..., gap] from the server (fst_check.split_words):
  // words at odd indices, numbered in order for П3's lookup.
  const p = el("p");
  pieces.forEach((piece, i) => {
    if (i % 2 === 1) p.append(el("span", { class: "w", "data-i": String((i - 1) / 2) }, piece));
    else if (piece) p.append(piece);
  });
  return p;
}

function nextBlock(next) {
  if (next == null) return el("p", { class: "muted next" }, "Это последняя глава.");
  if (next.status === "done") {
    return el(
      "div",
      { class: "next" },
      el("button", { onclick: () => showChapter(next.id) }, `Глава ${next.number} →`),
    );
  }
  return el(
    "p",
    { class: "muted next" },
    `Глава ${next.number}: ${STATUS_TEXT[next.status] || ""}. Бот напишет, когда она будет готова.`,
  );
}

async function showChapter(chapterId) {
  setBack(showBooks);
  let chapter;
  try {
    chapter = await api(`api/chapters/${encodeURIComponent(chapterId)}`);
  } catch (error) {
    showError(error);
    return;
  }
  show(
    el("p", { class: "muted small" }, chapter.book.title),
    el("h1", {}, `Глава ${chapter.number}`),
    el("p", { class: "muted small" }, chapterSubtitle(chapter)),
    el("div", { class: "banner" }, "Машинный перевод: изредка в нём бывают смысловые ошибки."),
    el("div", { class: "text", lang: "fi" }, ...chapter.paragraphs.map(paragraph)),
    nextBlock(chapter.next),
  );
}

if (tg) {
  tg.ready();
  tg.expand();
}
if (!tg || !tg.initData) show(el("p", { class: "muted" }, OPEN_FROM_BOT));
else showBooks();
