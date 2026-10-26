from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Protocol

from .agent import UltronAgent
from .llm_router import configured_provider_names

logger = logging.getLogger(__name__)

_BG = "#050505"
_SIDEBAR = "#0A0A0A"
_PANEL = "#0D0D0D"
_BORDER = "#252525"
_TEXT = "#F4F4F5"
_MUTED = "#969696"
_RED = "#F04444"
_DARK_RED = "#281010"


class ChatAgent(Protocol):
    def handle_task(self, message: str) -> str: ...

    def start_new_chat(self) -> None: ...


@dataclass
class DesktopChatSession:
    """Small UI-independent adapter around ULTRON's existing agent."""

    agent: ChatAgent

    def send(self, message: str) -> str:
        return self.agent.handle_task(message)

    def start_new_chat(self) -> None:
        self.agent.start_new_chat()

    def recent_history(self) -> list[dict[str, str]]:
        memory = getattr(self.agent, "memory", None)
        if memory is None:
            return []
        return memory.recent_conversation()


def ai_provider_status() -> str:
    names = configured_provider_names()
    return "AI · " + " → ".join(names).upper() if names else "AI PROVIDER NOT CONFIGURED"


def run_desktop_app(agent: UltronAgent | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable in this Python installation. "
            "Use the ULTRON CLI or install a Python distribution with Tk support."
        ) from exc

    session = DesktopChatSession(agent or UltronAgent())
    root = tk.Tk()
    root.title("ULTRON")
    root.geometry("1180x760")
    root.minsize(760, 560)
    root.configure(bg=_BG)

    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    closed = False
    busy = False
    thinking_frame = 0
    responses: queue.Queue[tuple[str, str]] = queue.Queue()
    visible_messages: list[tuple[str, str]] = []
    active_section = "Chat"
    sidebar_width = 228
    status_labels: list[object] = []
    status_dots: list[object] = []
    message_labels: list[object] = []

    def content_background(widget: object, color: str = _BG) -> None:
        try:
            widget.configure(bg=color)  # type: ignore[attr-defined]
        except (AttributeError, tk.TclError):
            pass

    def make_label(
        parent: object,
        text: str,
        *,
        color: str = _TEXT,
        size: int = 10,
        weight: str = "normal",
        background: str = _BG,
        **options: object,
    ) -> object:
        return tk.Label(
            parent,
            text=text,
            fg=color,
            bg=background,
            font=("Helvetica Neue", size, weight),
            **options,
        )

    def make_button(
        parent: object,
        text: str,
        command,
        *,
        accent: bool = False,
        active: bool = False,
        **options: object,
    ) -> object:
        background = _DARK_RED if active else (_RED if accent else _SIDEBAR)
        foreground = "#FFFFFF" if accent or active else "#D4D4D4"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            anchor="w",
            padx=14,
            pady=10,
            relief=tk.FLAT,
            bd=0,
            bg=background,
            fg=foreground,
            activebackground="#3A1717" if active or accent else "#171717",
            activeforeground="#FFFFFF",
            disabledforeground="#666666",
            font=("Helvetica Neue", 10, "bold" if accent or active else "normal"),
            cursor="hand2",
            **options,
        )
        if not accent and not active:
            button.bind("<Enter>", lambda _event: button.configure(bg="#151515"))
            button.bind("<Leave>", lambda _event: button.configure(bg=_SIDEBAR))
        return button

    sidebar = tk.Frame(root, bg=_SIDEBAR, width=sidebar_width)
    sidebar.grid(row=0, column=0, sticky="nsw")
    sidebar.grid_propagate(False)
    sidebar.grid_columnconfigure(0, weight=1)

    main = tk.Frame(root, bg=_BG)
    main.grid(row=0, column=1, sticky="nsew")
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=1)

    brand = tk.Frame(sidebar, bg=_SIDEBAR, padx=20, pady=22)
    brand.grid(row=0, column=0, sticky="ew")
    brand.grid_columnconfigure(1, weight=1)
    tk.Label(
        brand,
        text="U",
        fg=_RED,
        bg=_SIDEBAR,
        font=("Helvetica Neue", 24, "bold"),
        width=2,
    ).grid(row=0, column=0, rowspan=2, sticky="w")
    make_label(brand, "ULTRON", size=15, weight="bold", background=_SIDEBAR).grid(
        row=0, column=1, sticky="w"
    )
    make_label(brand, "PERSONAL AI", color=_MUTED, size=8, background=_SIDEBAR).grid(
        row=1, column=1, sticky="w"
    )

    nav = tk.Frame(sidebar, bg=_SIDEBAR, padx=12)
    nav.grid(row=1, column=0, sticky="new")
    nav.grid_columnconfigure(0, weight=1)
    section_buttons: dict[str, object] = {}
    status_var = tk.StringVar(value="")
    provider_var = tk.StringVar(value="")

    def set_status(text: str, *, error: bool = False) -> None:
        status_var.set(text)
        color = _RED if error else _MUTED
        for label in status_labels:
            try:
                label.configure(fg=color)
            except tk.TclError:
                pass
        for dot in status_dots:
            try:
                dot.configure(fg=color)
            except tk.TclError:
                pass

    def add_message(speaker: str, message: str, *, persist: bool = True) -> None:
        if persist:
            visible_messages.append((speaker, message))
            del visible_messages[:-100]
        if active_section == "Chat":
            render_messages()

    def clear_widgets(parent: object) -> None:
        for child in parent.winfo_children():  # type: ignore[attr-defined]
            child.destroy()

    def show_settings() -> None:
        popup = tk.Toplevel(root)
        popup.title("ULTRON Settings")
        popup.configure(bg=_PANEL)
        popup.resizable(False, False)
        popup.transient(root)
        popup.grab_set()
        card = tk.Frame(popup, bg=_PANEL, padx=26, pady=24)
        card.pack(fill=tk.BOTH, expand=True)
        make_label(card, "SETTINGS", size=12, weight="bold", background=_PANEL).pack(
            anchor="w", pady=(0, 18)
        )
        make_label(
            card, ai_provider_status(), background=_PANEL
        ).pack(anchor="w", pady=4)
        make_label(
            card, "Local commands remain available.", color=_MUTED, background=_PANEL
        ).pack(
            anchor="w", pady=4
        )
        workspace = str(getattr(session.agent, "base_dir", "Current directory"))
        make_label(
            card,
            f"Workspace  ·  {workspace}",
            color=_MUTED,
            background=_PANEL,
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=4)
        make_label(
            card,
            "API credentials are read from the process environment and are never shown here.",
            color=_MUTED,
            background=_PANEL,
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(14, 20))
        make_button(card, "Close", popup.destroy, accent=True).pack(anchor="e")

    def show_placeholder(section: str) -> None:
        nonlocal active_section
        active_section = section
        chat_widgets.clear()
        status_labels.clear()
        status_dots.clear()
        content_background(main)
        clear_widgets(main)
        main.grid_rowconfigure(1, weight=1)
        header = tk.Frame(main, bg=_BG, padx=28, pady=18)
        header.grid(row=0, column=0, sticky="ew")
        make_label(header, section.upper(), size=10, weight="bold").pack(side=tk.LEFT)
        make_button(header, "Settings", show_settings).pack(side=tk.RIGHT)
        body = tk.Frame(main, bg=_BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        card = tk.Frame(body, bg=_PANEL, highlightbackground=_BORDER, highlightthickness=1)
        card.place(relx=0.5, rely=0.46, relwidth=0.84, anchor="center", height=250)
        make_label(
            card, section.upper(), size=18, weight="bold", background=_PANEL
        ).pack(anchor="w", padx=28, pady=(30, 10))
        make_label(
            card,
            "This dedicated panel is coming soon.",
            color=_TEXT,
            size=12,
            background=_PANEL,
        ).pack(anchor="w", padx=28, pady=4)
        help_text = {
            "Projects": "Workspace actions are available through ULTRON chat.",
            "Memory": "Saved notes and local memory remain available through existing chat skills.",
            "Developer Tools": "Try inspect_project, code_helper, or run_project_check in chat.",
            "Security Lab": "Defensive local checks and security learning are available through chat.",
            "Game Development": "Try game plan or game note commands in chat.",
        }.get(section, "This panel is not connected to an operational feature yet.")
        make_label(
            card,
            help_text,
            color=_MUTED,
            size=10,
            background=_PANEL,
            wraplength=470,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=28, pady=(6, 22))
        make_button(card, "Back to chat", show_chat, accent=True).pack(
            anchor="w", padx=28
        )
        update_navigation()

    def show_history() -> None:
        nonlocal active_section
        active_section = "Chat History"
        chat_widgets.clear()
        status_labels.clear()
        status_dots.clear()
        clear_widgets(main)
        header = tk.Frame(main, bg=_BG, padx=28, pady=18)
        header.grid(row=0, column=0, sticky="ew")
        make_label(header, "CHAT HISTORY", size=10, weight="bold").pack(side=tk.LEFT)
        make_button(header, "Settings", show_settings).pack(side=tk.RIGHT)
        content = tk.Frame(main, bg=_BG, padx=28, pady=12)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        history = [
            item
            for item in session.recent_history()
            if not item["content"].startswith("[context] ")
        ]
        if not history:
            empty = tk.Frame(content, bg=_PANEL, padx=24, pady=24)
            empty.grid(row=0, column=0, sticky="nsew")
            make_label(empty, "No saved conversation yet.", color=_MUTED, background=_PANEL).pack(
                anchor="w"
            )
        else:
            history_scroll = tk.Scrollbar(
                content,
                orient=tk.VERTICAL,
                bg=_SIDEBAR,
                troughcolor=_BG,
                activebackground=_RED,
                highlightthickness=0,
                bd=0,
            )
            text = tk.Text(
                content,
                bg=_PANEL,
                fg=_TEXT,
                insertbackground=_TEXT,
                relief=tk.FLAT,
                wrap=tk.WORD,
                padx=18,
                pady=16,
                font=("Helvetica Neue", 10),
                state=tk.NORMAL,
                yscrollcommand=history_scroll.set,
            )
            history_scroll.configure(command=text.yview)
            text.grid(row=0, column=0, sticky="nsew")
            history_scroll.grid(row=0, column=1, sticky="ns")
            for turn in history:
                speaker = "YOU" if turn["role"] == "user" else "ULTRON"
                text.insert(tk.END, f"{speaker}\n{turn['content']}\n\n")
            text.configure(state=tk.DISABLED)
        footer = tk.Frame(content, bg=_BG)
        footer.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        make_button(footer, "Back to chat", show_chat, accent=True).pack(side=tk.LEFT)
        update_navigation()

    def on_nav(section: str) -> None:
        if section == "Chat":
            show_chat()
        elif section == "Chat History":
            show_history()
        elif section == "Settings":
            show_settings()
        else:
            show_placeholder(section)

    nav_items = (
        ("New Chat", "new"),
        ("Chat", "chat"),
        ("Chat History", "history"),
        ("Projects", "projects"),
        ("Memory", "memory"),
        ("Developer Tools", "developer"),
        ("Security Lab", "security"),
        ("Game Development", "games"),
    )
    for row, (label, _key) in enumerate(nav_items):
        if label == "New Chat":
            button = make_button(nav, label, lambda: new_chat(), accent=True)
        else:
            button = make_button(
                nav,
                label,
                lambda section=label: on_nav(section),
            )
            section_buttons[label] = button
        button.grid(row=row, column=0, sticky="ew", pady=3)

    sidebar_footer = tk.Frame(sidebar, bg=_SIDEBAR, padx=12, pady=14)
    sidebar_footer.grid(row=2, column=0, sticky="sew")
    sidebar.grid_rowconfigure(2, weight=1)
    settings_button = make_button(
        sidebar_footer, "Settings", show_settings, active=False
    )
    settings_button.pack(fill=tk.X)
    make_label(
        sidebar_footer,
        "LOCAL WORKSPACE",
        color="#626262",
        size=8,
        background=_SIDEBAR,
    ).pack(anchor="w", padx=14, pady=(12, 0))

    main.grid_rowconfigure(0, weight=0)
    main.grid_rowconfigure(1, weight=1)

    def update_navigation() -> None:
        for section, button in section_buttons.items():
            try:
                button.configure(
                    bg=_DARK_RED if section == active_section else _SIDEBAR,
                    fg=_TEXT if section == active_section else "#D4D4D4",
                )
            except tk.TclError:
                pass

    def scroll_to_bottom(_event: object | None = None) -> None:
        if "canvas" in chat_widgets:
            chat_widgets["canvas"].configure(
                scrollregion=chat_widgets["canvas"].bbox("all")
            )
            chat_widgets["canvas"].yview_moveto(1.0)

    chat_widgets: dict[str, object] = {}

    def render_messages() -> None:
        holder = chat_widgets.get("messages")
        if holder is None:
            return
        message_labels.clear()
        clear_widgets(holder)
        if not visible_messages:
            hero = tk.Frame(holder, bg=_BG)
            hero.pack(fill=tk.BOTH, expand=True, pady=(32, 22))
            make_label(hero, "ULTRON ONLINE", color=_TEXT, size=25, weight="bold").pack()
            make_label(
                hero,
                "Your personal AI. Your workspace. Your rules.",
                color=_MUTED,
                size=11,
            ).pack(pady=(8, 0))
            return
        for speaker, message in visible_messages[-100:]:
            is_user = speaker == "You"
            row = tk.Frame(holder, bg=_BG)
            row.pack(fill=tk.X, pady=7)
            alignment = "e" if is_user else "w"
            bubble_color = _DARK_RED if is_user else _PANEL
            bubble = tk.Frame(
                row,
                bg=bubble_color,
                highlightbackground="#5B2424" if is_user else _BORDER,
                highlightthickness=1,
                padx=14,
                pady=11,
            )
            bubble.pack(anchor=alignment, padx=(70, 0) if is_user else (0, 70))
            make_label(
                bubble,
                speaker.upper(),
                color=_RED if is_user else "#C7C7C7",
                size=8,
                weight="bold",
                background=bubble_color,
            ).pack(anchor="w")
            message_label = make_label(
                bubble,
                message,
                color=_TEXT,
                size=10,
                background=bubble_color,
                justify=tk.LEFT,
                wraplength=max(280, root.winfo_width() - sidebar_width - 170),
            )
            message_label.pack(anchor="w", pady=(5, 0))
            message_labels.append(message_label)
        scroll_to_bottom()

    def on_resize(event: object) -> None:
        if getattr(event, "widget", None) is not root:
            return
        wrap = max(280, root.winfo_width() - sidebar_width - 170)
        for label in message_labels:
            try:
                label.configure(wraplength=wrap)
            except tk.TclError:
                pass

    def send_message() -> None:
        nonlocal busy
        if busy:
            return
        input_widget = chat_widgets.get("input")
        if input_widget is None:
            return
        message = input_widget.get("1.0", tk.END).strip()
        if message.startswith("Message ULTRON..."):
            return
        if not message:
            return
        input_widget.delete("1.0", tk.END)
        add_message("You", message)
        busy = True
        send_widget = chat_widgets.get("send")
        if send_widget is not None:
            send_widget.configure(state=tk.DISABLED, text="WAIT")
        set_status("ULTRON IS THINKING")

        def handle() -> None:
            try:
                responses.put(("response", session.send(message)))
            except Exception as exc:
                logger.error("Desktop chat failed (%s).", type(exc).__name__)
                responses.put(
                    (
                        "error",
                        f"ULTRON could not complete that request ({type(exc).__name__}).",
                    )
                )

        threading.Thread(target=handle, name="ultron-chat", daemon=True).start()

    def show_attachment_notice() -> None:
        messagebox.showinfo(
            "Attachments",
            "File attachments are coming soon. You can still ask ULTRON about "
            "workspace files through chat.",
            parent=root,
        )

    def show_chat() -> None:
        nonlocal active_section
        active_section = "Chat"
        chat_widgets.clear()
        status_labels.clear()
        status_dots.clear()
        clear_widgets(main)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        top = tk.Frame(main, bg=_BG, padx=28, pady=14)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        left = tk.Frame(top, bg=_BG)
        left.grid(row=0, column=0, sticky="w")
        dot = tk.Label(
            left,
            text="●",
            fg=_MUTED,
            bg=_BG,
            font=("Helvetica Neue", 9),
        )
        dot.pack(
            side=tk.LEFT, padx=(0, 8)
        )
        status_dots.append(dot)
        status_widget = tk.Label(
            left,
            textvariable=status_var,
            fg=_MUTED,
            bg=_BG,
            font=("Helvetica Neue", 9, "bold"),
        )
        status_widget.pack(side=tk.LEFT)
        status_labels.append(status_widget)
        tk.Label(
            top,
            textvariable=provider_var,
            fg=_MUTED,
            bg=_BG,
            font=("Helvetica Neue", 9),
        ).grid(row=0, column=1, sticky="e", padx=(10, 18))
        make_button(top, "Settings", show_settings).grid(row=0, column=2, sticky="e")
        chat_area = tk.Frame(main, bg=_BG, padx=28, pady=18)
        chat_area.grid(row=1, column=0, sticky="nsew")
        chat_area.grid_columnconfigure(0, weight=1)
        chat_area.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(chat_area, bg=_BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            chat_area,
            orient=tk.VERTICAL,
            command=canvas.yview,
            bg=_SIDEBAR,
            troughcolor=_BG,
            activebackground=_RED,
            highlightthickness=0,
            bd=0,
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        message_holder = tk.Frame(canvas, bg=_BG)
        window_id = canvas.create_window(
            (0, 0), window=message_holder, anchor="nw", tags="messages"
        )
        message_holder.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        chat_widgets.update(
            canvas=canvas,
            messages=message_holder,
        )
        render_messages()

        composer = tk.Frame(
            main,
            bg=_PANEL,
            padx=14,
            pady=12,
            highlightbackground=_BORDER,
            highlightthickness=1,
        )
        composer.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 24))
        composer.grid_columnconfigure(1, weight=1)
        attach_button = tk.Button(
            composer,
            text="+",
            command=show_attachment_notice,
            bg=_PANEL,
            fg=_MUTED,
            activebackground=_PANEL,
            activeforeground=_RED,
            relief=tk.FLAT,
            bd=0,
            font=("Helvetica Neue", 18),
            cursor="hand2",
        )
        attach_button.grid(row=0, column=0, sticky="n", padx=(0, 10))
        input_box = tk.Text(
            composer,
            height=3,
            wrap=tk.WORD,
            bg=_PANEL,
            fg=_TEXT,
            insertbackground=_RED,
            selectbackground="#492020",
            relief=tk.FLAT,
            bd=0,
            padx=4,
            pady=6,
            font=("Helvetica Neue", 11),
        )
        input_box.grid(row=0, column=1, sticky="ew")
        input_box.insert(
            "1.0",
            "Message ULTRON... (Return for new line, ⌘/Ctrl + Return to send)",
        )
        placeholder_visible = True

        def focus_input(_event: object | None = None) -> None:
            nonlocal placeholder_visible
            if placeholder_visible:
                input_box.delete("1.0", tk.END)
                placeholder_visible = False
                input_box.configure(fg=_TEXT)

        input_box.bind("<FocusIn>", focus_input)
        input_box.bind("<Control-Return>", lambda _event: (send_message(), "break")[1])
        input_box.bind("<Command-Return>", lambda _event: (send_message(), "break")[1])
        send_button = tk.Button(
            composer,
            text="SEND  ↑",
            command=send_message,
            bg=_RED,
            fg="#FFFFFF",
            activebackground="#D83232",
            activeforeground="#FFFFFF",
            disabledforeground="#777777",
            relief=tk.FLAT,
            bd=0,
            padx=17,
            pady=11,
            font=("Helvetica Neue", 9, "bold"),
            cursor="hand2",
        )
        send_button.grid(row=0, column=2, sticky="se", padx=(10, 0))
        chat_widgets["input"] = input_box
        chat_widgets["send"] = send_button
        update_navigation()

    def new_chat() -> None:
        if busy:
            set_status("Wait for the current response before starting a new chat.")
            return
        if not messagebox.askyesno(
            "Start a new chat?",
            "This clears the saved recent conversation and cancels any pending "
            "action. Tasks and saved memories will remain.",
            parent=root,
        ):
            return
        try:
            session.start_new_chat()
        except Exception as exc:
            logger.error("Could not start a new chat (%s).", type(exc).__name__)
            messagebox.showerror(
                "Could not start a new chat",
                f"ULTRON could not reset this conversation ({type(exc).__name__}).",
                parent=root,
            )
            set_status("ERROR · could not reset chat", error=True)
            return
        visible_messages.clear()
        set_status("New chat ready")
        show_chat()
        chat_widgets["input"].focus_set()

    def poll_responses() -> None:
        nonlocal busy, thinking_frame
        if closed:
            return
        try:
            while True:
                kind, message = responses.get_nowait()
                if kind == "error":
                    add_message("Error", message)
                    set_status("ERROR · request failed", error=True)
                else:
                    add_message("ULTRON", message)
                    lowered = message.casefold()
                    if any(
                        phrase in lowered
                        for phrase in (
                            "provider request failed",
                            "task failed",
                            "confirmed task failed",
                            " failed:",
                        )
                    ):
                        set_status("ERROR · provider or task failure", error=True)
                    else:
                        set_status("LOCAL MODE")
                busy = False
                if chat_widgets.get("send") is not None:
                    try:
                        chat_widgets["send"].configure(state=tk.NORMAL, text="SEND  ↑")
                    except tk.TclError:
                        pass
        except queue.Empty:
            pass
        if busy:
            thinking_frame = (thinking_frame + 1) % 4
            set_status("ULTRON IS THINKING" + "." * thinking_frame)
        root.after(250, poll_responses)

    status_var.set(
        "LOCAL MODE"
        if not configured_provider_names()
        else "AI KEYS CONFIGURED · LIVE CONNECTION NOT VERIFIED"
    )
    provider_var.set(ai_provider_status())

    def close_app() -> None:
        nonlocal closed
        closed = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_app)
    root.bind("<Configure>", on_resize)
    show_chat()
    if visible_messages == []:
        visible_messages.extend(
            ("You" if turn["role"] == "user" else "ULTRON", turn["content"])
            for turn in session.recent_history()
            if not turn["content"].startswith("[context] ")
        )
        render_messages()
    root.after(250, poll_responses)
    root.mainloop()
