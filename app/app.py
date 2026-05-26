"""Streamlit demo for the geo assistant — B2B layout.

Layout:
    ┌──────────┬───────────────────────────────────┬──────────────────┐
    │          │            HEADER                 │                  │
    │ Sidebar  ├───────────────────────────────────┼──────────────────┤
    │ (state)  │                                   │ [hex card]       │
    │ read-    │       MAP (H3 hex grid)           │ CHAT             │
    │ only     │       click hex → card on right   │ multi-turn       │
    │          │                                   │                  │
    └──────────┴───────────────────────────────────┴──────────────────┘

Запуск:
    streamlit run app/app.py
"""
from __future__ import annotations
import re
import sys
import uuid
from pathlib import Path

import streamlit as st

# --- path bootstrap ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h3  # noqa: E402
from agent.agent import run as run_agent  # noqa: E402
from agent.db import init_db, load_chat_history, save_opportunity_grid, load_opportunity_grid  # noqa: E402
from agent.memory import PersistedMemory  # noqa: E402
from agent.prompts import SYSTEM_PROMPT  # noqa: E402

from core_utils.coverage import compute_opportunity_grid, HEX_SIZE_REFERENCE  # noqa: E402
from core_utils.search import search_places  # noqa: E402
from lib.data_types import Place, Hex  # noqa: E402
from map_visualization import opportunity_hex_map  # noqa: E402

from streamlit_folium import st_folium
import folium  # noqa: E402


# --- page config ------------------------------------------------------------
st.set_page_config(
    page_title="Гео ИИ-Ассистент — Выбор локации B2B",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }
      section[data-testid="stSidebar"] .stCaption { margin-top: -0.5rem; }
      div.chat-panel-marker { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- constants --------------------------------------------------------------
CITY_CENTER = (56.8386, 60.6055)
DEFAULT_ZOOM = 12
CATEGORIES = ["cafe", "restaurant", "fastfood", "pharmacy", "bar", "delivery"]
CATEGORY_LABELS: dict[str, str] = {
    "cafe": "кафе",
    "restaurant": "ресторан",
    "fastfood": "фастфуд",
    "pharmacy": "аптека",
    "bar": "бар",
    "delivery": "доставка",
}
DEFAULT_HEX_RES = 8


# --- DB-backed chat helpers -------------------------------------------------
def _load_chat_log_from_db(chat_id: str) -> list[dict]:
    """Load display-ready messages from DB (user + assistant text only)."""
    history = load_chat_history(chat_id)
    if not history:
        return []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant") and m.get("content")
    ]


def _restore_opportunity_grid(chat_id: str) -> None:
    """Restore the last opportunity_grid from the dedicated DB column into session_state."""
    grid_data = load_opportunity_grid(chat_id)
    if grid_data:
        grid_data['cells'] = [Hex(**c) for c in grid_data['cells']]
        st.session_state["opportunity_grid"] = grid_data


# --- session state initialization ------------------------------------------
def _init_state() -> None:
    init_db()  # Ensure table exists before any reads

    if "chat_id" not in st.session_state:
        url_chat_id = st.query_params.get("chat_id")
        if url_chat_id and load_chat_history(url_chat_id):
            # Restore existing session from URL
            chat_id = url_chat_id
        else:
            # New session — write id to URL so the user can bookmark it
            chat_id = str(uuid.uuid4())
            st.query_params["chat_id"] = chat_id
        st.session_state["chat_id"] = chat_id

    # Клик по hex_id в сообщении агента приходит как ?hex_click=...
    # Перехватываем, выставляем pinned_hex_id, чистим URL, делаем rerun.
    clicked_hex = st.query_params.get("hex_click")
    if clicked_hex:
        st.session_state["pinned_hex_id"] = clicked_hex
        del st.query_params["hex_click"]
        st.rerun()

    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = _load_chat_log_from_db(st.session_state["chat_id"])
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = "pharmacy"
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    if "selected_hex" not in st.session_state:
        st.session_state["selected_hex"] = None
    if "highlighted_hexes" not in st.session_state:
        st.session_state["highlighted_hexes"] = set()
    if "pinned_hex_id" not in st.session_state:
        st.session_state["pinned_hex_id"] = None
    if not st.session_state.get("opportunity_grid"):
        _restore_opportunity_grid(st.session_state["chat_id"])


_init_state()


# --- helpers ----------------------------------------------------------------
def _find_hex_by_click(lat: float, lng: float) -> Hex | None:
    """Найти гекс из текущей сетки по координатам клика.

    Используем тот же hex_resolution, что был при построении: считаем
    hex_id от координат клика и ищем совпадение в cells. Это надёжнее,
    чем искать «ближайший» по евклидову расстоянию.
    """
    opp = st.session_state.get("opportunity_grid")
    if not opp or not opp.get("cells"):
        return None

    args = opp.get("args", {})
    resolution = int(args.get("hex_resolution", DEFAULT_HEX_RES))

    try:
        target_hex_id = h3.latlng_to_cell(lat, lng, resolution)
    except Exception:
        return None

    for cell in opp["cells"]:
        if cell.hex_id == target_hex_id:
            return cell
    return None


def _format_hex_query(cell: Hex, category: str) -> str:
    """Сформировать обогащённый запрос агенту по метрикам гекса."""
    return (
        f"Analyse the area around hex {cell.hex_id} "
        f"(center: lat={cell.center_lat:.5f}, lon={cell.center_lon:.5f}) "
        f"for opening a new {category}.\n\n"
        f"Hex metrics:\n"
        f"- Opportunity score: {cell.opportunity_score:.2f}\n"
        f"- Demand (other POI): {cell.demand_score}\n"
        f"- Existing {category} competitors: {cell.competitor_count} "
        f"(avg rating {cell.competitor_avg_rating:.1f})\n"
        f"- Total POI in hex: {cell.total_places}\n\n"
        f"Call nearest_hexes(hex_id='{cell.hex_id}', radius=1) to get neighbourhood "
        f"context (includes row/col grid positions of each hex), "
        f"then nearest_places for real competitors near the hex centre, "
        f"and give a verdict on whether it is a good location."
    )


# --- sidebar (read-only agent state) ----------------------------------------
def render_sidebar() -> None:
    """Read-only дашборд: показывает текущий контекст агента."""
    with st.sidebar:
        st.header("Контекст агента")
        st.caption("Read-only — обновляется по мере диалога")

        # --- City ----------------------------------------------------------
        st.markdown("**Город**")
        st.markdown("Екатеринбург, Россия")

        st.divider()

        # --- Active analysis -----------------------------------------------
        opp = st.session_state.get("opportunity_grid")
        st.markdown("**Активный анализ**")

        if opp and opp.get("cells"):
            args = opp.get("args", {})
            category = args.get("category", "—")
            resolution = args.get("hex_resolution", DEFAULT_HEX_RES)
            threshold = args.get("demand_threshold", 0.0)
            n_cells = len(opp["cells"])
            n_visible = sum(1 for c in opp["cells"] if c.is_visible)

            st.markdown(f"**Сетка конкурентов** · `{CATEGORY_LABELS.get(category, category)}`")
            col_a, col_b = st.columns(2)
            col_a.metric("Всего гексов", n_cells)
            col_b.metric("Видимых", n_visible)

            st.caption(f"H3 resolution: **{resolution}**")
            hex_ref = HEX_SIZE_REFERENCE.get(resolution)
            if hex_ref:
                st.caption(f"↳ {hex_ref}")
            if threshold > 0:
                st.caption(f"Порог спроса: {threshold}")
        else:
            st.markdown("_Анализ ещё не запущен._")
            st.caption("Спросите агента или нажмите быстрое действие.")

        st.divider()

        # --- Heatmap state (если агент строил) ----------------------------
        hm = st.session_state.get("agent_heatmap")
        if hm:
            st.markdown("**Тепловая карта**")
            st.caption(f"Точек: {len(hm.get('points', []))}")
            st.divider()


        st.markdown("**Категории на карте**")
        st.markdown(f"🎯 `{CATEGORY_LABELS.get(st.session_state['selected_category'], st.session_state['selected_category'])}`")
        st.caption("Категория конкурентов на карте")

        # --- DEBUG: pinned_hex_id (можно удалить после проверки) -----------
        pinned_dbg = st.session_state.get("pinned_hex_id")
        st.caption(f"🔧 pinned: `{pinned_dbg}`")
        if pinned_dbg and opp and opp.get("cells"):
            in_grid = any(c.hex_id == pinned_dbg for c in opp["cells"])
            st.caption(f"🔧 в сетке: `{in_grid}`")


# --- map rendering ----------------------------------------------------------
def _build_map() -> folium.Map:
    """Собрать folium-карту из текущего состояния."""
    opp_config = st.session_state.get("opportunity_grid")

    if opp_config and opp_config.get("cells"):
        cells = opp_config["cells"]
        category = st.session_state["selected_category"]
        competitors = search_places(category=[category], limit=500)
        highlighted = st.session_state.get("highlighted_hexes") or None
        pinned = st.session_state.get("pinned_hex_id") or None

        strategy = opp_config.get("args", {}).get("strategy", "implant")
        if strategy == "aggregate":
            # Высокий agg-score = сильные конкуренты вокруг = красный.
            # Инвертируем палитру, метрика та же.
            color_kwargs: dict = {
                "color_metric": "opportunity_score",
                "colormap": "RdYlGn_r",
            }
        else:  # implant
            # Высокий opportunity_score = много клиентов гипотетическому
            # заведению = зелёный.
            color_kwargs = {
                "color_metric": "opportunity_score",
                "colormap": "RdYlGn",
            }

        fmap = opportunity_hex_map(
            cells,
            places=competitors,
            zoom_start=DEFAULT_ZOOM,
            highlighted_hex_ids=highlighted,
            pinned_hex_id=pinned,
            demand_threshold=50.0,
            **color_kwargs,
        )
        return fmap

    return folium.Map(location=CITY_CENTER, zoom_start=DEFAULT_ZOOM)


@st.fragment
def render_map_fragment() -> None:
    """Фрагмент с картой.

    Клик по гексу → определяем hex_id → сохраняем в selected_hex →
    rerun → правая колонка показывает карточку с метриками.
    """
    if st.session_state.get("highlighted_hexes"):
        if st.button("← Вся карта", key="reset_highlight"):
            st.session_state["highlighted_hexes"] = set()
            st.session_state["pinned_hex_id"] = None
            st.rerun()

    fmap = _build_map()

    # ВАЖНО про key у st_folium:
    # Компонент `st_folium` кэширует положение карты (центр/зум) между ререндерами
    # внутри одного и того же key. Если key не меняется, новые `location`/`zoom_start`,
    # переданные в folium.Map, игнорируются — карта остаётся там, где её последним
    # сдвинул пользователь. Поэтому мы включаем в key и `pinned`, и хэш набора
    # подсвеченных гексов: как только меняется любое из этих состояний (а
    # это ровно тот момент, когда мы хотим переехать на новый гекс), key
    # меняется → компонент пересоздаётся → подхватывает новый центр карты.
    pinned = st.session_state.get("pinned_hex_id") or "none"
    highlighted = st.session_state.get("highlighted_hexes") or set()
    highlighted_key = str(hash(frozenset(highlighted))) if highlighted else "all"
    map_state = st_folium(
        fmap,
        key=f"main_map_{pinned}_{highlighted_key}",
        height=600,
        use_container_width=True,
        returned_objects=["last_object_clicked", "last_clicked"],
    )

    clicked = map_state.get("last_object_clicked") if map_state else None
    if clicked and clicked != st.session_state.get("_last_click_handled"):
        lat = clicked.get("lat")
        lng = clicked.get("lng")
        if lat is not None and lng is not None:
            st.session_state["_last_click_handled"] = clicked
            cell = _find_hex_by_click(lat, lng)
            if cell is not None:
                st.session_state["selected_hex"] = cell
                # ВАЖНО: не сбрасываем pinned_hex_id здесь. Клик по карте
                # выбирает гекс в карточку (selected_hex), а pinned_hex_id —
                # это маркер из чата (контур на карте). Они независимы.
                st.rerun()
            # Если клик не попал в гекс (например, на маркер конкурента) —
            # тихо игнорируем. Можно добавить отдельный popup, но это уже сверху.


# --- chat panel -------------------------------------------------------------
def _send_to_agent(query: str) -> None:
    st.session_state.pop("highlighted_hexes", None)
    st.session_state["pinned_hex_id"] = None
    chat_id = st.session_state["chat_id"]
    try:
        run_agent(query, chat_id=chat_id)
    except Exception as exc:  # noqa: BLE001
        st.session_state["chat_log"].append({"role": "user", "content": query})
        st.session_state["chat_log"].append({"role": "assistant", "content": f"⚠️ Ошибка агента: {exc}"})
        return
    st.session_state["chat_log"] = _load_chat_log_from_db(chat_id)
    _restore_opportunity_grid(chat_id)


def render_hex_card() -> None:
    """Карточка с метриками выбранного гекса.

    Показывается над чатом, ждёт явного решения пользователя:
    отправить запрос агенту или отклонить.
    """
    cell = st.session_state.get("selected_hex")
    if cell is None:
        return

    cat = st.session_state["selected_category"]
    opp_score = cell.opportunity_score

    # Визуальная подсказка: зелёный — хорошее место, красный — плохое
    if opp_score > 5:
        verdict_emoji = "🟢"
        verdict_text = "Перспективная локация"
    elif opp_score > 0:
        verdict_emoji = "🟡"
        verdict_text = "Смешанные сигналы"
    else:
        verdict_emoji = "🔴"
        verdict_text = "Насыщено / слабо"

    with st.container(border=True):
        st.markdown(f"### {verdict_emoji} Выбранный гекс")
        st.caption(
            f"**{verdict_text}** для открытия `{CATEGORY_LABELS.get(cat, cat)}` · "
            f"`{cell.center_lat:.4f}, {cell.center_lon:.4f}`"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Возможность",
            f"{opp_score:.1f}",
            help="demand_score − competitor_density. Выше = лучше место.",
        )
        m2.metric(
            "Конкуренты",
            cell.competitor_count,
            help=f"Существующие {CATEGORY_LABELS.get(cat, cat)} в этом гексе",
        )
        m3.metric(
            "Всего мест",
            cell.total_places,
            help="Любые места — прокси трафика и жизни района",
        )

        if cell.competitor_count > 0:
            st.caption(
                f"Средний рейтинг конкурентов: "
                f"⭐ {cell.competitor_avg_rating:.1f}"
            )

        btn_ask, btn_dismiss = st.columns([2, 1])
        if btn_ask.button(
            "🔍 Спросить агента об этой зоне",
            use_container_width=True,
            type="primary",
            key="hex_card_ask",
        ):
            st.session_state["pending_query"] = _format_hex_query(cell, cat)
            st.session_state["selected_hex"] = None
            st.rerun()

        if btn_dismiss.button(
            "✕ Закрыть",
            use_container_width=True,
            key="hex_card_dismiss",
        ):
            st.session_state["selected_hex"] = None
            st.rerun()


_HEX_ID_RE = re.compile(r'\b([0-9a-f]{15})\b')


def _hex_button_label(hex_id: str) -> str:
    grid = st.session_state.get("opportunity_grid")
    if grid:
        for cell in grid.get("cells", []):
            if cell.hex_id == hex_id:
                return f"Гекс [{cell.label}]" if cell.label else f"{hex_id[:4]}…{hex_id[-4:]}"
    return f"{hex_id[:4]}…{hex_id[-4:]}"


# Injected once per page: MutationObserver that hides all buttons whose label
# is exactly a 15-char hex string (our hidden trigger buttons).
# УДАЛЕНО: больше не нужно — клики по hex_id обрабатываются через query_params,
# скрытые кнопки больше не создаются.


def _render_chat_message(msg: dict) -> None:
    content = msg["content"]
    if msg["role"] != "assistant":
        st.markdown(content)
        return

    # Если в сообщении нет hex_id — обычный markdown.
    if not _HEX_ID_RE.search(content):
        st.markdown(content)
        return

    chat_id = st.session_state.get("chat_id", "")

    # Превращаем 15-hex-id в обычные ссылки.
    # Ключевой момент: target="_top" — нативный HTML-механизм, который
    # инструктирует браузер открыть ссылку в top-level окне, а не в текущем
    # iframe. Работает в sandbox БЕЗ JS-навигации (которая блокируется).
    # Same-tab гарантирован — ни target="_blank", ни window.open() нет.
    def _make_hex_link(m: re.Match) -> str:
        hid = m.group(1)
        href = f"?chat_id={chat_id}&hex_click={hid}"
        return (
            f'<a href="{href}" target="_top" '
            f'title="{_hex_button_label(hid)}" '
            f'style="color:#FF8C00;font-family:monospace;background:#fff3e0;'
            f'padding:1px 5px;border-radius:3px;cursor:pointer;font-size:0.9em;'
            f'text-decoration:none">'
            f'{hid}</a>'
        )

    styled = _HEX_ID_RE.sub(_make_hex_link, content)
    st.markdown(styled, unsafe_allow_html=True)


def render_chat_panel() -> None:
    """Правая колонка: карточка гекса (если есть) + чат."""
    st.markdown('<div class="chat-panel-marker"></div>', unsafe_allow_html=True)

    # Карточка над чатом — самое видное место
    render_hex_card()

    # Контролы
    top_left, top_right = st.columns([3, 1.5])
    with top_left:
        st.session_state["selected_category"] = st.selectbox(
            "Целевая категория",
            CATEGORIES,
            index=CATEGORIES.index(st.session_state["selected_category"]),
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
            help="Категория конкурентов на карте и в анализе гексов.",
        )
    with top_right:
        st.write("")
        if st.button("🗑️ Очистить", use_container_width=True):
            # Start a brand-new session so the old one stays intact in DB
            new_chat_id = str(uuid.uuid4())
            st.session_state["chat_id"] = new_chat_id
            st.query_params["chat_id"] = new_chat_id
            st.session_state["chat_log"] = []
            st.session_state.pop("opportunity_grid", None)
            st.session_state.pop("agent_heatmap", None)
            st.session_state.pop("_last_click_handled", None)
            st.session_state.pop("highlighted_hexes", None)
            st.session_state["selected_hex"] = None
            st.session_state["pinned_hex_id"] = None
            st.rerun()

    with st.expander("Быстрые действия", expanded=False):
        b1, b2 = st.columns(2)
        if b1.button("📍 Показать сетку конкурентов", use_container_width=True):
            chat_id = st.session_state["chat_id"]
            cat = st.session_state["selected_category"]
            cells = compute_opportunity_grid(
                category=cat, hex_resolution=DEFAULT_HEX_RES, strategy="aggregate"
            )
            grid_data = {
                "cells": cells,
                "args": {
                    "category": cat,
                    "hex_resolution": DEFAULT_HEX_RES,
                    "demand_threshold": 0.0,
                    "strategy": "aggregate",
                },
            }
            st.session_state["opportunity_grid"] = grid_data
            save_opportunity_grid(chat_id, {'cells': [c.model_dump() for c in cells], 'args': grid_data['args']})
            msg_content = (
                f"Построена сетка конкурентов для **{CATEGORY_LABELS.get(cat, cat)}** "
                f"({len(cells)} гексов, "
                f"{sum(1 for c in cells if c.is_visible)} видимых). "
                f"Нажмите на гекс, чтобы увидеть метрики."
            )
            mem = PersistedMemory(chat_id=chat_id, system_prompt=SYSTEM_PROMPT)
            mem.add_assistant_message(msg_content)
            mem.save()
            st.session_state["chat_log"] = _load_chat_log_from_db(chat_id)
            st.rerun()

        if b2.button("❓ Где открыть?", use_container_width=True):
            cat = st.session_state["selected_category"]
            st.session_state["pending_query"] = (
                f"Где в городе лучше всего открыть новый {CATEGORY_LABELS.get(cat, cat)}? "
                f"Используй сетку возможностей, чтобы найти гексы "
                f"с высоким спросом и низкой конкуренцией."
            )
            st.rerun()

    st.divider()

    chat_container = st.container(height=380, border=False)
    with chat_container:
        if not st.session_state["chat_log"]:
            st.caption(
                "👋 Кликните на гекс на карте, используйте быстрые действия "
                "или просто задайте вопрос о городе."
            )
        for msg in st.session_state["chat_log"]:
            with st.chat_message(msg["role"]):
                _render_chat_message(msg)

    pending = st.session_state.pop("pending_query", None)
    if pending:
        with st.spinner("Думаю..."):
            _send_to_agent(pending)
        st.rerun()

    user_input = st.chat_input("Спросите о городе или кликните на гекс...")
    if user_input:
        with st.spinner("Думаю..."):
            _send_to_agent(user_input)
        st.rerun()


# --- layout -----------------------------------------------------------------
render_sidebar()

st.title("🗺️ Гео ИИ-Ассистент — Выбор локации B2B")
st.caption(
    "Анализ городской среды для открытия новой точки. "
    "Кликайте по гексам на карте — увидите метрики и сможете спросить агента."
)

map_col, chat_col = st.columns([3, 2], gap="medium")

with map_col:
    render_map_fragment()

with chat_col:
    render_chat_panel()
