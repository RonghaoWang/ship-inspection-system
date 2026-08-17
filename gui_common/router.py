"""页面路由与顶部面包屑导航。

设计要点：
- Router 管理页面栈（push/pop），维护面包屑
- BasePage 是所有页面基类，负责画顶栏和内容区容器
- 所有子页面只填 build_content()
"""
from __future__ import annotations

from typing import Callable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class BasePage(QWidget):
    """所有页面的基类。提供顶部面包屑、返回按钮、内容容器。"""

    def __init__(self, title_path: List[str], router: "Router | None" = None) -> None:
        super().__init__()
        self._title_path = title_path
        self._router = router
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏
        top = QFrame()
        top.setObjectName("topBar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 12, 0)
        top_layout.setSpacing(4)

        back = QPushButton("← 返回")
        back.setObjectName("backButton")
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self._on_back)
        top_layout.addWidget(back)

        breadcrumb = QLabel(" > ".join(self._title_path))
        breadcrumb.setObjectName("breadcrumb")
        top_layout.addWidget(breadcrumb)
        top_layout.addStretch(1)

        root.addWidget(top)

        # 内容容器
        content_wrap = QWidget()
        content_layout = QVBoxLayout(content_wrap)
        content_layout.setContentsMargins(32, 16, 32, 16)
        self._content_layout = content_layout
        root.addWidget(content_wrap, 1)

        self.build_content(content_layout)

    def build_content(self, layout: QVBoxLayout) -> None:  # pragma: no cover
        """子页面重写：往 layout 里塞内容。"""
        raise NotImplementedError

    def _on_back(self) -> None:
        if self._router is not None:
            self._router.pop()


class HomePage(QWidget):
    """首页与其他页面版式不同，不带面包屑。"""

    def __init__(self) -> None:
        super().__init__()

    def set_router(self, router: "Router") -> None:
        self._router = router


class Router(QStackedWidget):
    """页面栈路由。push 进入新页面、pop 回退。"""

    def __init__(self) -> None:
        super().__init__()
        self._factories: dict[str, Callable[[], QWidget]] = {}
        self._stack: List[str] = []

    def register(self, name: str, factory: Callable[[], QWidget]) -> None:
        self._factories[name] = factory

    def push(self, name: str) -> None:
        widget = self._factories[name]()
        if isinstance(widget, HomePage):
            widget.set_router(self)
        elif isinstance(widget, BasePage):
            widget._router = self
        idx = self.addWidget(widget)
        self._stack.append(name)
        self.setCurrentIndex(idx)

    def pop(self) -> None:
        if len(self._stack) <= 1:
            return
        current = self.currentWidget()
        self._stack.pop()
        # 回到上一个页面（仍在栈中的最后一个 widget）
        prev_idx = self.count() - 2
        self.setCurrentIndex(prev_idx)
        # 清理弹出的页面，避免堆积
        self.removeWidget(current)
        current.deleteLater()

    def reset_to(self, name: str) -> None:
        """清空栈，重置到某个页面（用于首页跳转）。"""
        while self.count() > 0:
            w = self.widget(0)
            self.removeWidget(w)
            w.deleteLater()
        self._stack = []
        self.push(name)
