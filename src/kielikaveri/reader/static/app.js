// The reader Mini App (plan 2026-11-20-kirjakaveri-reader, "П2", "П3"): the
// list of books and chapters, one chapter as paragraphs of word spans, and a
// bottom popup with the tapped word's lookup and 🔊.
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
const TRY_AGAIN = "Не получилось, попробуй ещё раз.";

class ApiError extends Error {
  constructor(status, reason) {
    super(`HTTP ${status}`);
    this.status = status;
    // A 503 from the reader's breaker says why: "rate" or "daily".
    this.reason = reason || null;
  }
}

// Every API call goes through here: initData rides in the Authorization
// header (the server checks it). With a body the call is a JSON POST.
async function request(path, { body, signal } = {}) {
  const options = {
    headers: { Authorization: "tma " + (tg ? tg.initData : "") },
    signal,
  };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    let reason = null;
    if (response.status === 503) {
      try {
        reason = (await response.json()).reason;
      } catch (_) {
        // No JSON body: shown as "rate".
      }
    }
    throw new ApiError(response.status, reason);
  }
  return response;
}

async function api(path, options) {
  return (await request(path, options)).json();
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

// What the BackButton does when no popup is open.
let pageBack = null;

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

function setPageBack(handler) {
  pageBack = handler;
  setBack(handler);
}

// --- list of books ------------------------------------------------------------

async function showBooks() {
  closePopup();
  setPageBack(null);
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

function paragraph(pieces, index) {
  // [gap, word, gap, word, ..., gap] from the server (fst_check.split_words):
  // words at odd indices. data-i is the word's index in
  // fst_check.words(paragraph), data-p the paragraph's index in the chapter:
  // together they are the address /api/lookup takes.
  const p = el("p", { "data-p": String(index) });
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

function textBlock(chapter) {
  const text = el("div", { class: "text", lang: "fi" }, ...chapter.paragraphs.map(paragraph));
  text.addEventListener("click", (event) => {
    const word = event.target.closest("span.w");
    if (word) openWord(chapter.id, word);
  });
  return text;
}

async function showChapter(chapterId) {
  closePopup();
  setPageBack(showBooks);
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
    textBlock(chapter),
    nextBlock(chapter.next),
  );
}

// --- word popup (П3) ----------------------------------------------------------

const popup = el("div", { class: "popup", role: "dialog", hidden: "" });
document.body.append(popup);

// The word the popup is about: {key, span, controller}. key is null after a
// failed lookup, so tapping the same word again retries.
let current = null;
let audioController = null;

function wordAddress(chapterId, span) {
  return {
    chapter_id: chapterId,
    paragraph: Number(span.closest("p").dataset.p),
    token: Number(span.dataset.i),
  };
}

function errorText(error) {
  if (error instanceof ApiError && error.status === 401) return OPEN_FROM_BOT;
  if (error instanceof ApiError && error.status === 503) {
    return error.reason === "daily"
      ? "Слишком много запросов, попробуй завтра."
      : "Слишком много запросов, попробуй через несколько минут.";
  }
  return TRY_AGAIN;
}

function openWord(chapterId, span) {
  const address = wordAddress(chapterId, span);
  const key = `${address.chapter_id}:${address.paragraph}:${address.token}`;
  // The same word again: its lookup is still running or already shown.
  if (current && current.key === key) return;
  // Another word: the previous lookup (and its sound) is aborted.
  closePopup();
  const mine = { key, span, controller: new AbortController() };
  current = mine;
  span.classList.add("active");
  popup.replaceChildren(el("p", { class: "muted" }, "загружаю…"));
  popup.hidden = false;
  document.body.classList.add("with-popup");
  setBack(closePopup);
  keepVisible(span);
  api("api/lookup", { body: address, signal: mine.controller.signal })
    .then((data) => {
      if (current === mine) renderLookup(data, address);
    })
    .catch((error) => {
      if (current !== mine || error.name === "AbortError") return;
      mine.key = null;
      popup.replaceChildren(el("p", { class: "muted" }, errorText(error)));
    });
}

function renderLookup(data, address) {
  const nodes = [
    el("p", { class: "lemma" },
      el("b", { lang: "fi" }, data.lemma),
      data.pos ? el("span", { class: "muted small" }, " " + data.pos) : null),
  ];
  if (data.puhekieli) {
    nodes.push(el("p", { class: "muted small" }, "разговорное: ",
      el("span", { lang: "fi" }, data.puhekieli)));
  }
  if (data.form_ru) nodes.push(el("p", { class: "form muted" }, data.form_ru));
  nodes.push(el("p", { class: "translation" }, data.translation_ru));
  if (data.sentence_ru) {
    nodes.push(el("p", { class: "sentence muted small" }, "в оригинале: ", data.sentence_ru));
  }
  if (data.fst_flagged || !data.in_dictionary) {
    nodes.push(el("p", { class: "warn small" },
      "⚠️ Словарь не знает это слово: может, это имя или ошибка перевода. Проверь."));
  }
  const speak = el("button", { class: "speak", type: "button", "aria-label": "Произнести" }, "🔊");
  const speakError = el("p", { class: "muted small" });
  speak.addEventListener("click", () => playWord(address, speak, speakError));
  nodes.push(el("div", { class: "actions" }, speak), speakError);
  popup.replaceChildren(...nodes);
  keepVisible(current.span);
}

// 🔊: the mp3 comes through fetch (an <audio src> can't send Authorization),
// plays from a blob: URL, and the URL is revoked once it has played.
async function playWord(address, button, errorNode) {
  if (button.disabled) return;
  button.disabled = true;
  errorNode.textContent = "";
  const controller = new AbortController();
  audioController = controller;
  let blob;
  try {
    blob = await (await request("api/tts", { body: address, signal: controller.signal })).blob();
  } catch (error) {
    if (error.name !== "AbortError") errorNode.textContent = errorText(error);
    button.disabled = false;
    return;
  }
  if (controller.signal.aborted) return;
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  const done = () => {
    URL.revokeObjectURL(url);
    button.disabled = false;
  };
  audio.addEventListener("ended", done, { once: true });
  audio.addEventListener("error", done, { once: true });
  try {
    await audio.play();
  } catch (_) {
    done();
    errorNode.textContent = TRY_AGAIN;
  }
}

function closePopup() {
  if (current) {
    current.controller.abort();
    current.span.classList.remove("active");
    current = null;
  }
  if (audioController) {
    audioController.abort();
    audioController = null;
  }
  if (popup.hidden) return;
  popup.hidden = true;
  popup.replaceChildren();
  document.body.classList.remove("with-popup");
  setBack(pageBack);
}

// Scroll the tapped word above the popup when the popup covers it.
function keepVisible(span) {
  const word = span.getBoundingClientRect();
  const top = popup.getBoundingClientRect().top;
  if (word.bottom > top - 8) {
    window.scrollBy({ top: word.bottom - top + 24, behavior: "smooth" });
  }
}

// A tap outside the popup (and not on another word) closes it.
document.addEventListener("click", (event) => {
  if (popup.hidden) return;
  if (popup.contains(event.target) || event.target.closest("span.w")) return;
  closePopup();
});

if (tg) {
  tg.ready();
  tg.expand();
}
if (!tg || !tg.initData) show(el("p", { class: "muted" }, OPEN_FROM_BOT));
else showBooks();
