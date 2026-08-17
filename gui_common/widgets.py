"""通用组件：卡片、按钮工厂、图像显示占位。"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def primary_button(text: str, on_click: Callable[[], None] | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("primaryButton")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn


def secondary_button(text: str, on_click: Callable[[], None] | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("secondaryButton")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn


def big_card(icon: str, title: str, desc: str, action_text: str, on_action: Callable[[], None]) -> QFrame:
    """首页 / 阶段页的大卡片。"""
    card = QFrame()
    card.setObjectName("card")
    card.setMinimumSize(320, 260)
    layout = QVBoxLayout(card)
    layout.setSpacing(6)
    layout.setContentsMargins(24, 24, 24, 24)

    icon_lbl = QLabel(icon)
    icon_lbl.setObjectName("cardIcon")
    icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(icon_lbl)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("cardTitle")
    title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title_lbl)

    desc_lbl = QLabel(desc)
    desc_lbl.setObjectName("cardDesc")
    desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    desc_lbl.setWordWrap(True)
    layout.addWidget(desc_lbl)

    layout.addStretch(1)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(primary_button(action_text, on_action))
    btn_row.addStretch(1)
    layout.addLayout(btn_row)

    return card


def image_display(placeholder: str = "请选择输入") -> QLabel:
    from PySide6.QtWidgets import QSizePolicy
    lbl = QLabel(placeholder)
    lbl.setObjectName("imageDisplay")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setMinimumSize(360, 280)
    lbl.setScaledContents(False)
    # 主动占满可用空间；同时忽略 pixmap 内容对 sizeHint 的影响
    policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    policy.setHeightForWidth(False)
    lbl.setSizePolicy(policy)
    return lbl


def big_result(value: str, label: str) -> QWidget:
    """结果面板中央的大数字：42%、0.83 m² 等。"""
    wrap = QWidget()
    layout = QVBoxLayout(wrap)
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)

    val = QLabel(value)
    val.setObjectName("resultBig")
    val.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(val)

    lbl = QLabel(label)
    lbl.setObjectName("resultLabel")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(lbl)

    return wrap
