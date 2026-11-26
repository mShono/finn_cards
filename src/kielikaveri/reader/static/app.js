// The reader Mini App (plan 2026-11-20-kirjakaveri-reader, "П2"-"П6"):
// the list of books and chapters, one chapter as paragraphs of word spans, a
// bottom popup with the tapped word's lookup, 🔊 and ➕, the learning
// screens (flashcards rated by buttons) and the decks (lists, search, a
// word's card with its edits).
// No build step. Book and card text is only ever set through textContent.
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
  constructor(status, body) {
    super(`HTTP ${status}`);
    this.status = status;
    // The JSON body's reason, if any: a 503 from the reader's breaker says
    // "rate" or "daily", a 422 from /api/cards may say "save_failed", a 409
    // from /api/learn "stale", "session_gone" or "nothing_to_undo".
    this.reason = (body && body.reason) || null;
    // The whole JSON body: a 409 "stale" carries the session's current state.
    this.body = body || null;
  }
}

// Every API call goes through here: initData rides in the Authorization
// header (the server checks it). With a body the call is a JSON POST unless
// method says otherwise (PATCH, DELETE: the decks, П6).
async function request(path, { method, body, signal } = {}) {
  const options = {
    method: method || (body !== undefined ? "POST" : "GET"),
    headers: { Authorization: "tma " + (tg ? tg.initData : "") },
    signal,
  };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    let body = null;
    try {
      body = await response.json();
    } catch (_) {
      // No JSON body (aiohttp's own 401/404 pages): only the status counts.
    }
    throw new ApiError(response.status, body && typeof body === "object" ? body : null);
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

// The flashcard on screen (П5), or null: show() drops it, so keys and late
// answers can tell a card that is gone.
let cardView = null;

function show(...nodes) {
  cardView = null;
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

// Bumped by every move to another screen. A request remembers the value it
// started under and drops its answer if the user has moved on since.
let navigation = 0;

function navigate(back) {
  closePopup();
  navigation += 1;
  setPageBack(back);
  return navigation;
}

// "📖 Книги | 📚 Учить | 🗂 Колоды" on top of the three root screens.
function switcher(active) {
  const tab = (name, label, go) =>
    el(
      "button",
      {
        type: "button",
        class: name === active ? "tab active" : "tab",
        "aria-pressed": String(name === active),
        onclick: () => {
          if (name !== active) go();
        },
      },
      label,
    );
  return el(
    "nav",
    { class: "switch" },
    tab("books", "📖 Книги", showBooks),
    tab("learn", "📚 Учить", showLearn),
    tab("decks", "🗂 Колоды", showDeckList),
  );
}

// --- list of books ------------------------------------------------------------

async function showBooks() {
  const view = navigate(null);
  let data;
  try {
    data = await api("api/books");
  } catch (error) {
    if (view === navigation) showError(error);
    return;
  }
  if (view !== navigation) return;
  if (data.books.length === 0) {
    show(
      switcher("books"),
      el("h1", {}, "Книг пока нет"),
      el("p", { class: "muted" }, "Пришли книгу боту файлом .txt - переведу её на финский."),
    );
    return;
  }
  const nodes = [switcher("books")];
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

// back: where the BackButton goes - the books, or the chapter's deck (П6).
async function showChapter(chapterId, back = showBooks) {
  const view = navigate(back);
  let chapter;
  try {
    chapter = await api(`api/chapters/${encodeURIComponent(chapterId)}`);
  } catch (error) {
    if (view === navigation) showError(error);
    return;
  }
  if (view !== navigation) return;
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

// The word the popup is about: {key, span, controller, looked}. key is null
// after a failed lookup, so tapping the same word again retries. looked is set
// once the lookup has answered or failed: a later /api/lemma is then ignored.
// The controller aborts both the lemma and the lookup request.
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
  if (isConflict(error, "session_gone")) return SESSION_GONE_TEXT;
  // TTS without an OpenAI key: no sound, everything else works.
  if (error instanceof ApiError && error.status === 503 && error.reason === "no_openai") {
    return "Озвучка сейчас недоступна.";
  }
  if (error instanceof ApiError && error.status === 503) {
    return error.reason === "daily"
      ? "Слишком много запросов, попробуй завтра."
      : "Слишком много запросов, попробуй через несколько минут.";
  }
  if (error instanceof ApiError && error.status === 422 && error.reason === "save_failed") {
    return "Не получилось сохранить карточку.";
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
  const mine = { key, span, controller: new AbortController(), looked: false };
  current = mine;
  span.classList.add("active");
  popup.replaceChildren(el("p", { class: "muted" }, "загружаю…"));
  popup.hidden = false;
  document.body.classList.add("with-popup");
  setBack(closePopup);
  keepVisible(span);
  // Both at once: the FST lemma answers in milliseconds and fills the first
  // line while the lookup (a model call, seconds) is still running.
  api("api/lemma", { body: address, signal: mine.controller.signal })
    .then((data) => {
      if (current === mine && !mine.looked) renderLemma(data);
    })
    .catch(() => {
      // Silently: the lookup shows the lemma, or its own error, anyway.
    });
  api("api/lookup", { body: address, signal: mine.controller.signal })
    .then((data) => {
      if (current !== mine) return;
      mine.looked = true;
      renderLookup(data, address);
    })
    .catch((error) => {
      if (current !== mine || error.name === "AbortError") return;
      mine.looked = true;
      mine.key = null;
      popup.replaceChildren(el("p", { class: "muted" }, errorText(error)));
    });
}

// The popup's first line before the lookup answers: the FST lemma (several
// readings joined by " / "). No lemma: the popup stays at "загружаю…".
function renderLemma(data) {
  const lemmas = [...new Set((data.lemmas || []).filter(Boolean))];
  if (lemmas.length === 0) return;
  popup.replaceChildren(
    el("p", { class: "lemma" }, el("b", { lang: "fi" }, lemmas.join(" / "))),
    el("p", { class: "muted" }, "загружаю…"),
  );
  keepVisible(current.span);
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
  speak.addEventListener("click", () =>
    playSound(speak, speakError, (signal) => request("api/tts", { body: address, signal })));
  const add = el("button", { class: "add", type: "button", "aria-label": "В колоду" }, "➕");
  const addError = el("p", { class: "muted small" });
  if (data.in_deck) markAdded(add, "duplicate");
  else add.addEventListener("click", () => addCard(address, add, addError));
  nodes.push(el("div", { class: "actions" }, speak, add), speakError, addError);
  popup.replaceChildren(...nodes);
  keepVisible(current.span);
}

// ➕: saves the word as a card in the chapter's deck. The request is not
// aborted when the popup closes (the card is saved or it is not). Its answer
// only touches this button and this error line: they belong to the popup they
// were made for and are detached once another word is shown.
function addCard(address, button, errorNode) {
  if (button.disabled) return;
  button.disabled = true;
  button.textContent = "…";
  errorNode.textContent = "";
  api("api/cards", { body: address })
    .then((data) => markAdded(button, data.status))
    .catch((error) => {
      button.disabled = false;
      button.textContent = "➕";
      errorNode.textContent = errorText(error);
    });
}

function markAdded(button, status) {
  button.disabled = true;
  button.classList.add("done");
  button.textContent = status === "duplicate" ? "✓ уже в колоде" : "✓ в колоде";
}

// 🔊: the WAV comes through fetch (an <audio src> can't send Authorization)
// and plays through Web Audio. An Audio element on a blob: URL, played after
// an awaited fetch, stayed silent in Telegram (Android and web): the tap's user activation is
// gone by then. An AudioContext resumed synchronously inside the tap stays
// unlocked, so the sound may start whenever the fetch ends.
let audioContext = null;
let audioSource = null;

function unlockAudio() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) return null;
  if (!audioContext) audioContext = new Context();
  // Must run in the tap itself, before any await.
  if (audioContext.state !== "running") audioContext.resume().catch(() => {});
  return audioContext;
}

function stopSound() {
  if (audioSource) {
    try {
      audioSource.stop();
    } catch (_) {
      // Already stopped.
    }
    audioSource = null;
  }
}

function soundErrorText(error) {
  // The error's name only (NotAllowedError, EncodingError, ...): it helps to
  // tell an autoplay block from a broken file, and carries no book text.
  return error && error.name ? `Не получилось воспроизвести (${error.name}).` : TRY_AGAIN;
}

// fetchSound(signal) makes the request: the word's /api/tts in the popup,
// the card example's /api/learn/tts on a flashcard.
function playSound(button, errorNode, fetchSound) {
  if (button.disabled) return;
  const context = unlockAudio();
  if (!context) {
    errorNode.textContent = "Этот браузер не умеет воспроизводить звук.";
    return;
  }
  button.disabled = true;
  errorNode.textContent = "";
  stopSound();
  const controller = new AbortController();
  audioController = controller;
  fetchAndPlay(context, fetchSound, controller, button, errorNode);
}

async function fetchAndPlay(context, fetchSound, controller, button, errorNode) {
  let data;
  try {
    const response = await fetchSound(controller.signal);
    data = await response.arrayBuffer();
  } catch (error) {
    if (error.name !== "AbortError") errorNode.textContent = errorText(error);
    button.disabled = false;
    return;
  }
  if (controller.signal.aborted) return;
  try {
    // The callback form: older Safari has no promise-returning decodeAudioData.
    const buffer = await new Promise((resolve, reject) =>
      context.decodeAudioData(data, resolve, reject));
    if (controller.signal.aborted) return;
    // The tap's resume() may still be settling: wait for it, but at most a
    // second - when the browser refused the unlock it can stay pending forever.
    if (context.state !== "running") {
      await Promise.race([
        context.resume().catch(() => {}),
        new Promise((resolve) => setTimeout(resolve, 1000)),
      ]);
    }
    if (controller.signal.aborted) return;
    if (context.state !== "running") {
      button.disabled = false;
      errorNode.textContent = `Не получилось воспроизвести (AudioContext: ${context.state}).`;
      return;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => {
      if (audioSource === source) audioSource = null;
      button.disabled = false;
    };
    audioSource = source;
    source.start();
  } catch (error) {
    button.disabled = false;
    errorNode.textContent = soundErrorText(error);
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
  stopSound();
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

// --- learning (П5) ------------------------------------------------------------
// Screens: deck (only with more than one deck) -> side -> debt (if the
// server asks) -> card -> end. The session's queue lives on the server; the
// page holds one card at a time.

const SIDES = [
  ["fi", "🇫🇮 Финский"],
  ["ru", "🇷🇺 Русский"],
  ["mix", "🔀 Вперемешку"],
];
// FSRS ratings: "✗ Забыл" is Again (not Hard: Hard still counts as
// remembered), "✓ Помню" is Good. Hard and Easy are small buttons.
const RATING = { again: 1, hard: 2, good: 3, easy: 4 };
const SESSION_GONE_TEXT = "Сессия прервалась.";

// deckId: null means all decks. session: the server's id of the session.
// fromDeck: the deck screen "📚 Учить эту колоду" was tapped on (П6), where
// the side choice goes back to; null when learning started from the tab.
const learn = { decks: [], deckId: null, side: null, session: null, fromDeck: null };

function loading() {
  return el("p", { class: "muted" }, "загружаю…");
}

function isConflict(error, reason) {
  return error instanceof ApiError && error.status === 409 && error.reason === reason;
}

// The session cannot go on: replaced or lost on the server (409
// session_gone), or its card is gone (404: deleted after a lost answer).
function isSessionLost(error) {
  return isConflict(error, "session_gone") || (error instanceof ApiError && error.status === 404);
}

function showLearnError(error) {
  if (isConflict(error, "session_gone")) {
    showSessionGone();
    return;
  }
  navigate(learn.side ? showSides : null);
  show(el("p", { class: "muted" }, errorText(error)));
}

async function showLearn() {
  const view = navigate(null);
  learn.fromDeck = null;
  show(switcher("learn"), loading());
  let data;
  try {
    data = await api("api/learn/decks");
  } catch (error) {
    if (view === navigation) showLearnError(error);
    return;
  }
  if (view !== navigation) return;
  learn.decks = data.decks || [];
  if (learn.decks.length > 1) {
    showDecks();
  } else {
    // One deck (or none) needs no choice.
    learn.deckId = learn.decks.length === 1 ? learn.decks[0].id : null;
    showSides();
  }
}

function showDecks() {
  navigate(null);
  const list = el("ul", { class: "chapters" });
  const item = (label, deckId) =>
    el("li", { class: "ready", onclick: () => pickDeck(deckId) }, el("span", {}, label));
  for (const deck of learn.decks) list.append(item(deck.name, deck.id));
  list.append(item("Все", null));
  show(switcher("learn"), el("h1", {}, "Какую колоду?"), list);
}

function pickDeck(deckId) {
  learn.deckId = deckId;
  showSides();
}

function showSides() {
  // From a deck screen back goes to that deck. Otherwise the deck picker is
  // one step back only when there was a choice.
  const fromDeck = learn.fromDeck;
  navigate(fromDeck ? () => showDeck(fromDeck) : learn.decks.length > 1 ? showDecks : null);
  const buttons = SIDES.map(([side, label]) =>
    el("button", { type: "button", onclick: () => startSession(side) }, label));
  show(
    switcher("learn"),
    el("h1", {}, "Какой стороной?"),
    el("p", { class: "muted small" }, "🇫🇮 - финское слово, вспомни перевод. 🇷🇺 - наоборот."),
    el("div", { class: "choices" }, ...buttons),
  );
}

async function startSession(side) {
  learn.side = side;
  learn.session = null;
  const view = navigate(showSides);
  show(loading());
  let data;
  try {
    data = await api("api/learn/start", { body: { deck_id: learn.deckId, side } });
  } catch (error) {
    if (view !== navigation) return;
    // The deck was deleted after the list loaded: load the decks again.
    if (error instanceof ApiError && error.status === 404) showLearn();
    else showLearnError(error);
    return;
  }
  if (view === navigation) showState(data);
}

// Every learning answer is one of these states.
function showState(data) {
  if (data.session) learn.session = data.session;
  if (data.state === "card") showCard(data);
  else if (data.state === "debt") showDebt(data);
  else if (data.state === "end") showEnd(data);
  else showMessage(data.text || "Сейчас нечего повторять.");
}

function showMessage(text) {
  navigate(showSides);
  show(
    el("p", { class: "message" }, text),
    el("button", { type: "button", class: "secondary", onclick: showSides }, "Назад"),
  );
}

function showSessionGone() {
  navigate(showSides);
  show(
    el("p", { class: "message" }, SESSION_GONE_TEXT),
    el("button", { type: "button", onclick: () => startSession(learn.side) }, "Начать заново"),
  );
}

function showDebt(data) {
  const view = navigate(showSides);
  const errorNode = el("p", { class: "muted small" });
  const buttons = [];
  const choose = (action) => {
    if (buttons.some((button) => button.disabled)) return;
    for (const button of buttons) button.disabled = true;
    errorNode.textContent = "";
    api("api/learn/debt", { body: { session: learn.session, action } })
      .then((answer) => {
        if (view === navigation) showState(answer);
      })
      .catch((error) => {
        if (view !== navigation) return;
        if (isSessionLost(error)) {
          showSessionGone();
          return;
        }
        for (const button of buttons) button.disabled = false;
        errorNode.textContent = errorText(error);
      });
  };
  buttons.push(
    el("button", { type: "button", onclick: () => choose("batch") }, "Заниматься как обычно"),
    el("button", { type: "button", class: "secondary", onclick: () => choose("defer") },
      "Отложить остальное"),
  );
  show(
    el("p", { class: "message" },
      `Просрочено ${data.overdue} карточек - это много за одну сессию.\n` +
        `Разгребать как обычно (по ${data.max_cards} за раз) ` +
        `или отложить остальное на ${data.postpone_days} дн.?`),
    el("div", { class: "choices" }, ...buttons),
    errorNode,
  );
}

function showEnd(data) {
  navigate(showSides);
  const errorNode = el("p", { class: "muted small" });
  const nodes = [
    el("p", { class: "message" }, data.text || `Сессия окончена: ${data.reviewed} пройдено.`),
  ];
  if (data.can_undo) {
    const undo = el("button", { type: "button", class: "undo" }, "↩ Отменить");
    undo.addEventListener("click", () => undoLast(undo, errorNode, null));
    nodes.push(undo);
  }
  nodes.push(
    errorNode,
    el("div", { class: "choices" },
      el("button", { type: "button", onclick: showSides }, "Ещё сессия"),
      el("button", { type: "button", class: "secondary", onclick: showBooks }, "📖 Книги")),
  );
  show(...nodes);
}

// The two faces of a card, by its type (plan, "Карточка в ответе"). front:
// the prompt; answer: the main line of the back; label: the form's name.
function cardText(card) {
  if (card.type === "production") {
    return { flag: "🇷🇺", front: card.translation_ru, frontLang: "ru", answer: card.lemma, answerLang: "fi" };
  }
  if (card.type === "inflection") {
    // A form with no task: the lemma on both sides.
    if (!card.cue || !card.form_answer) {
      return { flag: "🇫🇮", front: card.lemma, frontLang: "fi", answer: card.lemma, answerLang: "fi" };
    }
    return {
      flag: "🇫🇮",
      front: `${card.lemma} → ${card.cue}`,
      frontLang: "fi",
      answer: card.form_answer,
      answerLang: "fi",
      label: card.form_label ? `✅ ${card.form_label}` : null,
    };
  }
  return { flag: "🇫🇮", front: card.lemma, frontLang: "fi", answer: card.translation_ru, answerLang: "ru" };
}

function showCard(data) {
  const view = navigate(showSides);
  const card = data.card;
  const text = cardText(card);
  const ui = { view, card, revealed: false, flipped: false, busy: false, buttons: [] };

  const done = (data.progress && data.progress.done) || 0;
  const total = (data.progress && data.progress.total) || 0;
  const fill = el("div", { class: "progress-fill" });
  fill.style.width = `${total ? Math.round((100 * done) / total) : 0}%`;
  const progress = el("div", { class: "learn-progress" },
    el("div", { class: "progress-bar" }, fill),
    el("span", { class: "muted small" }, `${done} / ${total}`));

  const front = el("div", { class: "face front" },
    el("p", { class: "flag" }, text.flag),
    el("p", { class: "prompt", lang: text.frontLang }, text.front));
  const back = el("div", { class: "face back" },
    el("p", { class: "muted small", lang: text.frontLang }, text.front),
    el("p", { class: "answer", lang: text.answerLang }, text.answer),
    text.label ? el("p", { class: "muted small", lang: "fi" }, text.label) : null);
  if (card.example_fi) back.append(el("p", { class: "example", lang: "fi" }, card.example_fi));
  if (card.example_ru) back.append(el("p", { class: "example muted" }, card.example_ru));
  const speakError = el("p", { class: "muted small" });
  if (card.example_fi) {
    const speak = el("button", { class: "speak", type: "button", "aria-label": "Произнести пример" }, "🔊");
    speak.addEventListener("click", () =>
      playSound(speak, speakError, (signal) =>
        request("api/learn/tts", { body: { session: learn.session, card_id: card.id }, signal })
          .catch((error) => {
            // The sound's 409 session_gone ends the session like a rating's.
            if (isConflict(error, "session_gone") && view === navigation) showSessionGone();
            throw error;
          })));
    back.append(speak, speakError);
  }
  ui.inner = el("div", { class: "flashcard-inner" }, front, back);
  ui.node = el("div", { class: "flashcard", role: "button", tabindex: "0",
    "aria-label": "Карточка: нажми, чтобы перевернуть" }, ui.inner);
  // A tap flips; 🔊 on the back is its own button.
  ui.node.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    flip(ui);
  });

  const rateButton = (label, rating, cls) => {
    const button = el("button", { type: "button", class: cls }, label);
    button.addEventListener("click", () => rate(ui, rating));
    ui.buttons.push(button);
    return button;
  };
  ui.rateArea = el("div", { class: "rate", hidden: "" },
    rateButton("✗ Забыл", RATING.again, "again"),
    rateButton("✓ Помню", RATING.good, "good"),
    rateButton("Трудно", RATING.hard, "fine"),
    rateButton("Легко", RATING.easy, "fine"));
  ui.hint = el("p", { class: "muted small hint" }, "Нажми на карточку, чтобы увидеть ответ.");
  ui.error = el("p", { class: "muted small" });

  // ↩ on a line of its own under the card and the rating buttons.
  const nodes = [progress, ui.node, ui.hint, ui.rateArea];
  if (data.can_undo) {
    const undo = el("button", { type: "button", class: "undo" }, "↩ Отменить");
    undo.addEventListener("click", () => undoLast(undo, ui.error, ui));
    ui.buttons.push(undo);
    nodes.push(undo);
  }
  nodes.push(ui.error);
  show(...nodes);
  cardView = ui;
}

function flip(ui) {
  if (ui.busy || ui !== cardView) return;
  ui.flipped = !ui.flipped;
  ui.revealed = true;
  ui.inner.classList.toggle("flipped", ui.flipped);
  ui.rateArea.hidden = false;
  ui.hint.hidden = true;
}

function setBusy(ui, busy) {
  ui.busy = busy;
  for (const button of ui.buttons) button.disabled = busy;
}

// One rating per card: busy is set before the request leaves, so a second
// key or button press while it is in flight sends nothing.
function rate(ui, rating) {
  if (ui.busy || !ui.revealed || ui !== cardView) return;
  setBusy(ui, true);
  ui.error.textContent = "";
  api("api/learn/review", { body: { session: learn.session, card_id: ui.card.id, rating } })
    .then((data) => {
      if (ui.view === navigation) showState(data);
    })
    .catch((error) => {
      if (ui.view !== navigation) return;
      // Stale: this card was rated already (another tab, a lost answer).
      // The body is the session's current state: redraw it, no message.
      if (isConflict(error, "stale") && error.body && error.body.state) {
        showState(error.body);
        return;
      }
      if (isSessionLost(error)) {
        showSessionGone();
        return;
      }
      setBusy(ui, false);
      ui.error.textContent = errorText(error);
    });
}

// ↩: on the next card (ui is its view) and on the end screen (ui is null).
function undoLast(button, errorNode, ui) {
  if (button.disabled || (ui && ui.busy)) return;
  if (ui) setBusy(ui, true);
  else button.disabled = true;
  errorNode.textContent = "";
  const view = navigation;
  api("api/learn/undo", { body: { session: learn.session } })
    .then((data) => {
      if (view === navigation) showState(data);
    })
    .catch((error) => {
      if (view !== navigation) return;
      if (isSessionLost(error)) {
        showSessionGone();
        return;
      }
      if (ui) setBusy(ui, false);
      else button.disabled = false;
      if (isConflict(error, "nothing_to_undo")) button.hidden = true;
      else errorNode.textContent = errorText(error);
    });
}

// --- decks (П6) ---------------------------------------------------------------
// Screens: the deck list (and the search over all decks) -> a deck -> a
// word. Nothing is cached: every screen asks the server again, so the lists
// and counters are fresh after an edit, a move or a delete.

const NOTES_PAGE = 50;
const SEARCH_DELAY_MS = 250;
const DECK_NAME_MAX = 64;
const MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
// A card's line on the word screen; an inflection card shows its form.
const CARD_SIDE = { recognition: "🇫🇮→🇷🇺", production: "🇷🇺→🇫🇮" };
const DECK_GONE_TEXT = "Колода уже удалена.";
const NOTE_GONE_TEXT = "Слово уже удалено.";
const DECK_NAME_TEXT = `Название - от 1 до ${DECK_NAME_MAX} символов.`;
const EMPTY_VALUE_TEXT = "Пустое значение не сохраню - напиши текст ещё раз.";

function isNotFound(error) {
  return error instanceof ApiError && error.status === 404;
}

// One request at a time on a screen: from the tap until the answer every
// action button of the screen is disabled, as the rating buttons are.
function makeGuard() {
  return { busy: false, buttons: [] };
}

function guardButton(guard, button) {
  guard.buttons.push(button);
  return button;
}

function setGuard(guard, busy) {
  guard.busy = busy;
  for (const button of guard.buttons) button.disabled = busy;
}

// tg.showConfirm (Telegram 6.2+), else the browser's confirm. A second tap
// while the dialog is open asks nothing: Telegram throws on a second popup.
let confirmOpen = false;

function askConfirm(text, onYes) {
  if (confirmOpen) return;
  const telegramConfirm =
    tg && typeof tg.showConfirm === "function" && (!tg.isVersionAtLeast || tg.isVersionAtLeast("6.2"));
  if (telegramConfirm) {
    confirmOpen = true;
    try {
      tg.showConfirm(text, (ok) => {
        confirmOpen = false;
        if (ok) onYes();
      });
      return;
    } catch (_) {
      confirmOpen = false;
    }
  }
  if (window.confirm(text)) onYes();
}

// A screen with one message and a way back: a deck or word deleted
// elsewhere, or a load that failed.
function showNotice(text, back) {
  navigate(back);
  show(
    el("p", { class: "message" }, text),
    el("button", { type: "button", class: "secondary", onclick: back }, "Назад"),
  );
}

function deckCounters(deck) {
  return `слов ${deck.notes} · к повторению ${deck.due}`;
}

function deckCaption(chapter) {
  return `📖 ${chapter.book_title}, гл. ${chapter.number}`;
}

function searchField(placeholder, value) {
  const input = el("input", {
    type: "search",
    class: "search",
    placeholder,
    autocomplete: "off",
    enterkeyhint: "search",
    "aria-label": placeholder,
  });
  input.value = value;
  return input;
}

// Asks /api/notes/search a moment after the last keystroke. An answer to an
// older query, or one that comes after the user has left the screen, is
// dropped. on.results(notes, q), on.empty(), on.error(error).
function liveSearch(input, view, on) {
  let timer = null;
  let asked = 0;
  const run = () => {
    clearTimeout(timer);
    asked += 1;
    const mine = asked;
    const q = input.value.trim();
    if (!q) {
      on.empty();
      return;
    }
    timer = setTimeout(() => {
      api(`api/notes/search?q=${encodeURIComponent(q)}`)
        .then((data) => {
          if (mine === asked && view === navigation) on.results(data.notes || [], q);
        })
        .catch((error) => {
          if (mine === asked && view === navigation) on.error(error);
        });
    }, SEARCH_DELAY_MS);
  };
  input.addEventListener("input", run);
  return run;
}

// "лемма — перевод"; in the search the deck's name under it (null: a note
// from before decks, in no deck).
function noteItem(note, back, withDeck) {
  const text = el("span", {}, el("b", { lang: "fi" }, note.lemma), ` — ${note.translation_ru}`);
  if (withDeck) {
    text.append(el("br"), el("span", { class: "muted small" },
      note.deck ? `🗂 ${note.deck.name}` : "без колоды"));
  }
  return el("li", { class: "ready", onclick: () => showNote(note.id, back) }, text);
}

// An input with "Сохранить" and "Отмена" under it and the error line under
// those. onSubmit(value, editor) gets the trimmed value; while its request
// runs the screen's guard keeps every button disabled and the save button
// reads "…".
function inlineEditor(guard, { value = "", placeholder = "", maxLength = 0, label = "Сохранить",
  onSubmit, onCancel }) {
  const input = el("input", { type: "text", class: "field", placeholder, autocomplete: "off",
    "aria-label": placeholder || label });
  if (maxLength) input.maxLength = maxLength;
  input.value = value;
  const save = guardButton(guard, el("button", { type: "button" }, label));
  const cancel = guardButton(guard, el("button", { type: "button", class: "secondary" }, "Отмена"));
  const error = el("p", { class: "error small" });
  const editor = { input, save, error, label, guard };
  const submit = () => {
    if (!guard.busy) onSubmit(input.value.trim(), editor);
  };
  save.addEventListener("click", submit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submit();
    }
  });
  cancel.addEventListener("click", () => {
    if (!guard.busy) onCancel();
  });
  editor.node = el("div", { class: "editor" }, input, el("div", { class: "pair" }, save, cancel), error);
  return editor;
}

function editorBusy(editor, busy) {
  setGuard(editor.guard, busy);
  editor.input.readOnly = busy;
  editor.save.textContent = busy ? "…" : editor.label;
  if (busy) editor.error.textContent = "";
}

// One edit from an editor: send() makes the request, done() runs on
// success (it always moves on: the screen is asked for again), a failure
// goes under the field through errorFor(error).
function submitEdit(editor, view, send, done, errorFor) {
  if (editor.guard.busy) return;
  editorBusy(editor, true);
  send()
    .then((data) => {
      if (view === navigation) done(data);
    })
    .catch((error) => {
      if (view !== navigation) return;
      editorBusy(editor, false);
      editor.error.textContent = errorFor(error);
    });
}

function deckNameErrorText(error) {
  if (error instanceof ApiError && error.status === 422) return DECK_NAME_TEXT;
  if (isNotFound(error)) return DECK_GONE_TEXT;
  return errorText(error);
}

// --- the deck list ------------------------------------------------------------

async function showDeckList() {
  const view = navigate(null);
  show(switcher("decks"), loading());
  let data;
  try {
    data = await api("api/decks");
  } catch (error) {
    if (view === navigation) show(switcher("decks"), el("p", { class: "muted" }, errorText(error)));
    return;
  }
  if (view !== navigation) return;
  const search = searchField("Поиск по всем колодам", "");
  // The first letter typed moves on to the search screen with it.
  search.addEventListener("input", () => {
    if (search.value.trim()) showSearch(search.value);
  });
  const list = el("ul", { class: "chapters decks" });
  for (const deck of data.decks) list.append(deckItem(deck));
  show(switcher("decks"), search, list, newDeckBlock(view));
}

function deckItem(deck) {
  const text = el("span", {}, deck.name, el("br"), el("span", { class: "muted small" }, deckCounters(deck)));
  if (deck.chapter) text.append(el("br"), el("span", { class: "muted small" }, deckCaption(deck.chapter)));
  return el("li", { class: "ready", onclick: () => showDeck(deck.id) }, text);
}

// "➕ Новая колода": the name is typed right under the list.
function newDeckBlock(view) {
  const block = el("div", { class: "new-deck" });
  const open = el("button", { type: "button", class: "secondary" }, "➕ Новая колода");
  open.addEventListener("click", () => {
    const editor = inlineEditor(makeGuard(), {
      placeholder: "Название колоды",
      maxLength: DECK_NAME_MAX,
      label: "Создать",
      onCancel: () => block.replaceChildren(open),
      onSubmit: (name, ed) => {
        if (!name) {
          ed.error.textContent = DECK_NAME_TEXT;
          return;
        }
        submitEdit(ed, view, () => api("api/decks", { body: { name } }), () => showDeckList(),
          deckNameErrorText);
      },
    });
    block.replaceChildren(editor.node);
    editor.input.focus();
  });
  block.append(open);
  return block;
}

// --- search over all decks ----------------------------------------------------

// focus: false when coming back from a word - the keyboard would cover the
// results.
function showSearch(query, focus = true) {
  const view = navigate(showDeckList);
  const input = searchField("Поиск по всем колодам", query);
  const status = el("p", { class: "muted small" });
  const list = el("ul", { class: "chapters words" });
  const run = liveSearch(input, view, {
    results: (notes, q) => {
      list.replaceChildren(...notes.map((note) => noteItem(note, () => showSearch(q, false), true)));
      status.textContent = notes.length ? "" : "Ничего не нашла.";
    },
    empty: () => {
      list.replaceChildren();
      status.textContent = "Слово по-фински или по-русски, во всех колодах.";
    },
    error: (error) => {
      list.replaceChildren();
      status.textContent = errorText(error);
    },
  });
  show(el("h1", {}, "Поиск"), input, status, list);
  if (focus) {
    input.focus();
    try {
      input.setSelectionRange(input.value.length, input.value.length);
    } catch (_) {
      // An input type without a caret API: the focus is enough.
    }
  }
  run();
}

// --- one deck -----------------------------------------------------------------

async function showDeck(deckId) {
  const view = navigate(showDeckList);
  show(loading());
  let decks;
  let page;
  try {
    // The counters come with the deck list (there is no one-deck GET).
    [decks, page] = await Promise.all([
      api("api/decks"),
      api(`api/decks/${encodeURIComponent(deckId)}/notes?offset=0&limit=${NOTES_PAGE}`),
    ]);
  } catch (error) {
    if (view !== navigation) return;
    showNotice(isNotFound(error) ? DECK_GONE_TEXT : errorText(error), showDeckList);
    return;
  }
  if (view !== navigation) return;
  const deck = decks.decks.find((item) => item.id === deckId);
  if (!deck) {
    showNotice(DECK_GONE_TEXT, showDeckList);
    return;
  }
  renderDeck(view, deck, page);
}

function renderDeck(view, deck, page) {
  const guard = makeGuard();
  const back = () => showDeck(deck.id);
  const errorNode = el("p", { class: "error small" });
  const editArea = el("div", {});

  const learnButton = guardButton(guard, el("button", { type: "button" }, "📚 Учить эту колоду"));
  learnButton.addEventListener("click", () => learnDeck(deck));
  const menuButton = guardButton(guard, el("button", { type: "button", class: "secondary menu-button",
    "aria-label": "Ещё", "aria-expanded": "false" }, "⋯"));
  const rename = guardButton(guard, el("button", { type: "button", class: "secondary" }, "✍️ Переименовать"));
  const remove = guardButton(guard, el("button", { type: "button", class: "secondary danger" },
    "🗑 Удалить колоду"));
  const menu = el("div", { class: "menu", hidden: "" }, rename, remove);
  menuButton.addEventListener("click", () => {
    menu.hidden = !menu.hidden;
    menuButton.setAttribute("aria-expanded", String(!menu.hidden));
  });
  rename.addEventListener("click", () => {
    menu.hidden = true;
    const editor = inlineEditor(guard, {
      value: deck.name,
      placeholder: "Название колоды",
      maxLength: DECK_NAME_MAX,
      onCancel: () => editArea.replaceChildren(),
      onSubmit: (name, ed) => {
        if (!name) {
          ed.error.textContent = DECK_NAME_TEXT;
          return;
        }
        submitEdit(ed, view,
          () => api(`api/decks/${encodeURIComponent(deck.id)}`, { method: "PATCH", body: { name } }),
          back, deckNameErrorText);
      },
    });
    editArea.replaceChildren(editor.node);
    editor.input.focus();
  });
  remove.addEventListener("click", () => deleteDeck(view, deck, guard, errorNode));

  const nodes = [el("h1", {}, deck.name)];
  if (deck.chapter) nodes.push(el("p", { class: "muted small" }, deckCaption(deck.chapter)));
  nodes.push(
    el("p", { class: "muted small counters" }, deckCounters(deck), el("br"),
      `формы: изучается ${deck.introduced}, ещё не открыто ${deck.not_introduced}`),
    el("div", { class: "deck-head" }, learnButton, menuButton),
    menu,
    editArea,
    errorNode,
  );
  // chapter.id is null-safe: the book's deletion leaves chapter null.
  if (deck.chapter && deck.chapter.id) {
    const chapterId = deck.chapter.id;
    nodes.push(el("button", { type: "button", class: "secondary chapter-button",
      onclick: () => showChapter(chapterId, back) }, "📖 Открыть главу"));
  }
  nodes.push(...deckWords(view, deck, page, back));
  show(...nodes);
}

function learnDeck(deck) {
  learn.decks = [];
  learn.deckId = deck.id;
  learn.fromDeck = deck.id;
  showSides();
}

function deleteDeck(view, deck, guard, errorNode) {
  if (guard.busy) return;
  askConfirm(
    `Удалить колоду «${deck.name}»? Слова (${deck.notes}) и история повторений удалятся вместе с ней.`,
    () => {
      if (guard.busy || view !== navigation) return;
      setGuard(guard, true);
      errorNode.textContent = "";
      request(`api/decks/${encodeURIComponent(deck.id)}`, { method: "DELETE" })
        .then(() => {
          if (view === navigation) showDeckList();
        })
        .catch((error) => {
          if (view !== navigation) return;
          // Deleted already (another tap, another tab): the list shows that.
          if (isNotFound(error)) {
            showDeckList();
            return;
          }
          setGuard(guard, false);
          errorNode.textContent = errorText(error);
        });
    },
  );
}

// The deck's words, newest first, 50 at a time: the next page loads when the
// "Ещё" button under the list scrolls into view (or is tapped).
// The search in the deck asks /api/notes/search and keeps this deck's notes:
// it finds words not loaded yet, folds case the way the server does (ä,
// Cyrillic), and costs one request per pause in typing. It sees what the
// server's search returns (at most 50 over all decks), enough for one user.
function deckWords(view, deck, page, back) {
  const list = el("ul", { class: "chapters words" });
  const found = el("ul", { class: "chapters words", hidden: "" });
  const status = el("p", { class: "muted small" });
  const more = el("button", { type: "button", class: "secondary more" }, "Ещё");
  const pager = { loaded: 0, total: page.total, loading: false };

  const append = (notes) => {
    for (const note of notes) list.append(noteItem(note, back, false));
    pager.loaded += notes.length;
    // An empty page means the deck shrank meanwhile: nothing more to ask.
    if (notes.length === 0) pager.total = pager.loaded;
    more.hidden = pager.loaded >= pager.total;
  };
  append(page.notes);
  if (pager.total === 0) status.textContent = "Слов пока нет.";

  let observer = null;
  const loadMore = () => {
    if (pager.loading || pager.loaded >= pager.total) return;
    pager.loading = true;
    more.disabled = true;
    more.textContent = "загружаю…";
    api(`api/decks/${encodeURIComponent(deck.id)}/notes?offset=${pager.loaded}&limit=${NOTES_PAGE}`)
      .then((data) => {
        if (view !== navigation) return;
        pager.total = data.total;
        append(data.notes);
        status.textContent = "";
      })
      .catch((error) => {
        if (view !== navigation) return;
        status.textContent = errorText(error);
      })
      .finally(() => {
        pager.loading = false;
        more.disabled = false;
        more.textContent = "Ещё";
        // Still in view after a short page: observe anew, which reports it.
        if (observer && !more.hidden && view === navigation) {
          observer.unobserve(more);
          observer.observe(more);
        }
      });
  };
  more.addEventListener("click", loadMore);
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver((entries) => {
      // Another screen: the button left the page, which reports here too.
      if (view !== navigation) {
        observer.disconnect();
        return;
      }
      if (entries.some((entry) => entry.isIntersecting)) loadMore();
    }, { rootMargin: "200px" });
    observer.observe(more);
  }

  const search = searchField("Поиск в колоде", "");
  liveSearch(search, view, {
    results: (notes) => {
      const mine = notes.filter((note) => note.deck && note.deck.id === deck.id);
      found.replaceChildren(...mine.map((note) => noteItem(note, back, false)));
      found.hidden = false;
      list.hidden = true;
      more.hidden = true;
      status.textContent = mine.length ? "" : "В этой колоде не нашла.";
    },
    empty: () => {
      found.hidden = true;
      found.replaceChildren();
      list.hidden = false;
      more.hidden = pager.loaded >= pager.total;
      status.textContent = pager.total === 0 ? "Слов пока нет." : "";
    },
    error: (error) => {
      status.textContent = errorText(error);
    },
  });
  return [pager.total > 0 ? search : null, status, list, found, more].filter(Boolean);
}

// --- one word -----------------------------------------------------------------

// back: the deck or the search the word was opened from.
async function showNote(noteId, back) {
  const view = navigate(back);
  show(loading());
  let note;
  try {
    note = await api(`api/notes/${encodeURIComponent(noteId)}`);
  } catch (error) {
    if (view === navigation) showNotice(isNotFound(error) ? NOTE_GONE_TEXT : errorText(error), back);
    return;
  }
  if (view === navigation) renderNote(view, note, back);
}

// "🇫🇮→🇷🇺: следующее повторение 3 дек, повторений 5, забыто 1".
function cardLine(card, now) {
  const side = CARD_SIDE[card.type] || `🔤 ${card.form || "форма"}`;
  if (card.status === "not_introduced") return `${side}: ещё не открыта`;
  const parts = [];
  const due = card.due ? new Date(card.due) : null;
  if (due && !Number.isNaN(due.getTime())) {
    parts.push(due <= now ? "пора повторить" : `следующее повторение ${shortDate(due, now)}`);
  }
  parts.push(`повторений ${card.reps || 0}`, `забыто ${card.lapses || 0}`);
  return `${side}: ${parts.join(", ")}`;
}

function shortDate(date, now) {
  const text = `${date.getDate()} ${MONTHS[date.getMonth()]}`;
  return date.getFullYear() === now.getFullYear() ? text : `${text} ${date.getFullYear()}`;
}

function noteEditErrorText(error, value) {
  // The bot's text for the same clash (bot/edit.py _clash_reply).
  if (isConflict(error, "duplicate")) {
    return `«${value}» уже есть в этой колоде отдельной карточкой - сначала удали одну из них.`;
  }
  if (error instanceof ApiError && error.status === 503 && error.reason === "no_openai") {
    return "Слово сейчас не поменять: формы пересчитывает OpenAI, а он не подключён.";
  }
  if (error instanceof ApiError && error.status === 502) {
    return "OpenAI не ответил, слово не изменилось. Попробуй ещё раз.";
  }
  if (isNotFound(error)) return NOTE_GONE_TEXT;
  if (error instanceof ApiError && error.status === 422) return EMPTY_VALUE_TEXT;
  return errorText(error);
}

function renderNote(view, note, back) {
  const guard = makeGuard();
  const reopen = () => showNote(note.id, back);
  const speakError = el("p", { class: "muted small" });
  const speakButton = (what, label) => {
    const button = el("button", { type: "button", class: "speak", "aria-label": label }, "🔊");
    button.addEventListener("click", () =>
      playSound(button, speakError, (signal) =>
        request(`api/notes/${encodeURIComponent(note.id)}/tts`, { body: { what }, signal })));
    return button;
  };

  const nodes = [
    el("div", { class: "word-head" }, el("h1", { lang: "fi" }, note.lemma),
      speakButton("lemma", "Произнести слово")),
  ];
  if (note.pos) nodes.push(el("p", { class: "muted small" }, note.pos));
  if (note.puhekieli) {
    nodes.push(el("p", { class: "muted small" }, "разговорное: ", el("span", { lang: "fi" }, note.puhekieli)));
  }
  nodes.push(el("p", { class: "translation" }, note.translation_ru));
  if (note.example_fi) {
    nodes.push(el("div", { class: "example-line" },
      el("p", { lang: "fi" }, note.example_fi), speakButton("example", "Произнести пример")));
  }
  if (note.example_ru) nodes.push(el("p", { class: "muted" }, note.example_ru));
  nodes.push(speakError);

  const forms = note.forms_verified && note.forms ? Object.entries(note.forms) : [];
  if (forms.length) {
    nodes.push(el("h2", {}, "Формы"), el("ul", { class: "forms" },
      ...forms.map(([key, form]) => el("li", {}, el("span", { class: "muted" }, key), " ",
        el("span", { lang: "fi" }, form)))));
  }

  const deckLine = el("p", { class: "muted small" }, note.deck ? `🗂 ${note.deck.name}` : "🗂 без колоды");
  if (note.deck && note.deck.chapter) deckLine.append(el("br"), deckCaption(note.deck.chapter));
  nodes.push(deckLine);

  const now = new Date();
  const cards = note.cards || [];
  nodes.push(el("h2", {}, "Карточки"), cards.length
    ? el("ul", { class: "card-states" }, ...cards.map((card) => el("li", {}, cardLine(card, now))))
    : el("p", { class: "muted small" }, "Карточек нет."));

  const editArea = el("div", {});
  const errorNode = el("p", { class: "error small" });
  const closeEdit = () => editArea.replaceChildren();
  const editField = (field, value, placeholder) => {
    const editor = inlineEditor(guard, {
      value,
      placeholder,
      onCancel: closeEdit,
      onSubmit: (text, ed) => {
        if (!text) {
          ed.error.textContent = EMPTY_VALUE_TEXT;
          return;
        }
        if (text === value) {
          closeEdit();
          return;
        }
        // ✍️ Слово is an LLM call (the forms): "…" until it answers.
        submitEdit(ed, view,
          () => api(`api/notes/${encodeURIComponent(note.id)}`, { method: "PATCH", body: { [field]: text } }),
          reopen, (error) => noteEditErrorText(error, text));
      },
    });
    errorNode.textContent = "";
    editArea.replaceChildren(editor.node);
    editor.input.focus();
  };
  const action = (label, onTap, cls = "secondary") => {
    const button = guardButton(guard, el("button", { type: "button", class: cls }, label));
    button.addEventListener("click", () => {
      if (!guard.busy) onTap();
    });
    return button;
  };
  nodes.push(
    el("div", { class: "word-actions" },
      action("✍️ Слово", () => editField("lemma", note.lemma, "Слово по-фински")),
      action("✍️ Перевод", () => editField("translation_ru", note.translation_ru, "Перевод")),
      action("📂 Перенести", () => pickTargetDeck(view, note, guard, editArea, errorNode, reopen)),
      action("🗑 Удалить", () => deleteNote(view, note, guard, errorNode, back), "secondary danger")),
    editArea,
    errorNode,
  );
  show(el("div", { class: "word" }, ...nodes));
}

// "📂 Перенести": the decks as buttons, this word's deck among them marked.
function pickTargetDeck(view, note, guard, editArea, errorNode, reopen) {
  setGuard(guard, true);
  errorNode.textContent = "";
  editArea.replaceChildren(loading());
  api("api/decks")
    .then((data) => {
      if (view !== navigation) return;
      setGuard(guard, false);
      const current = note.deck ? note.deck.id : null;
      const buttons = data.decks.map((deck) => {
        const here = deck.id === current;
        const button = el("button", { type: "button", class: "secondary" },
          here ? `${deck.name} (сейчас здесь)` : deck.name);
        if (here) button.disabled = true;
        else guardButton(guard, button);
        button.addEventListener("click", () => moveNote(view, note, deck, guard, errorNode, reopen));
        return button;
      });
      const cancel = guardButton(guard, el("button", { type: "button", class: "secondary" }, "Отмена"));
      cancel.addEventListener("click", () => {
        if (!guard.busy) editArea.replaceChildren();
      });
      editArea.replaceChildren(el("h2", {}, "В какую колоду?"), el("div", { class: "choices" }, ...buttons, cancel));
    })
    .catch((error) => {
      if (view !== navigation) return;
      setGuard(guard, false);
      editArea.replaceChildren();
      errorNode.textContent = errorText(error);
    });
}

function moveNote(view, note, deck, guard, errorNode, reopen) {
  if (guard.busy) return;
  setGuard(guard, true);
  errorNode.textContent = "";
  api(`api/notes/${encodeURIComponent(note.id)}/move`, { body: { deck_id: deck.id } })
    .then(() => {
      if (view === navigation) reopen();
    })
    .catch((error) => {
      if (view !== navigation) return;
      setGuard(guard, false);
      if (isConflict(error, "duplicate_in_target")) {
        errorNode.textContent = `В колоде «${deck.name}» это слово уже есть.`;
      } else if (isNotFound(error)) {
        errorNode.textContent = "Слово или колода уже удалены.";
      } else {
        errorNode.textContent = errorText(error);
      }
    });
}

function deleteNote(view, note, guard, errorNode, back) {
  if (guard.busy) return;
  askConfirm(`Удалить «${note.lemma}» со всеми карточками и историей повторений?`, () => {
    if (guard.busy || view !== navigation) return;
    setGuard(guard, true);
    errorNode.textContent = "";
    request(`api/notes/${encodeURIComponent(note.id)}`, { method: "DELETE" })
      .then(() => {
        if (view === navigation) back();
      })
      .catch((error) => {
        if (view !== navigation) return;
        // A second delete (double tap, another tab) finds nothing: 404.
        if (isNotFound(error)) {
          showNotice(NOTE_GONE_TEXT, back);
          return;
        }
        setGuard(guard, false);
        errorNode.textContent = errorText(error);
      });
  });
}

// web.telegram.org on a desktop: ← Forgot, → Remember, space flips.
document.addEventListener("keydown", (event) => {
  const ui = cardView;
  if (!ui || event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
  // Space or Enter on a focused button presses that button.
  if (event.target instanceof Element && event.target.closest("button")) return;
  if (event.key === " ") {
    event.preventDefault();
    flip(ui);
  } else if (event.key === "ArrowLeft" && ui.revealed) {
    event.preventDefault();
    rate(ui, RATING.again);
  } else if (event.key === "ArrowRight" && ui.revealed) {
    event.preventDefault();
    rate(ui, RATING.good);
  }
});

if (tg) {
  tg.ready();
  tg.expand();
}
// reader_url?screen=learn opens the learning screens, ?screen=decks the
// decks, ?screen=search the search (the bot's /delete). A query parameter,
// not the hash: Telegram keeps tgWebAppData there.
const screen = new URLSearchParams(window.location.search).get("screen");
if (!tg || !tg.initData) show(el("p", { class: "muted" }, OPEN_FROM_BOT));
else if (screen === "learn") showLearn();
else if (screen === "decks") showDeckList();
else if (screen === "search") showSearch("");
else showBooks();
