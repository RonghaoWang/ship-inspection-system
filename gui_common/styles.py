

PRIMARY = "#1a73e8"        # 主蓝
PRIMARY_HOVER = "#1557b0"
PRIMARY_LIGHT = "#e8f0fe"
TEXT_PRIMARY = "#202124"
TEXT_SECONDARY = "#5f6368"
BORDER = "#dadce0"
BG = "#ffffff"
BG_ALT = "#f8f9fa"
DANGER = "#d93025"
SUCCESS = "#188038"
WARNING = "#f29900"


STYLE_SHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background: {BG};
}}

/* 顶栏 */
QFrame#topBar {{
    background: {BG_ALT};
    border-bottom: 1px solid {BORDER};
    min-height: 40px;
    max-height: 40px;
}}

QLabel#breadcrumb {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    padding: 0 12px;
}}

QPushButton#backButton {{
    background: transparent;
    color: {PRIMARY};
    border: none;
    padding: 4px 10px;
    font-weight: 600;
}}
QPushButton#backButton:hover {{
    background: {PRIMARY_LIGHT};
    border-radius: 3px;
}}

/* 首页大标题 */
QLabel#homeTitle {{
    font-size: 32px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 24px 0 8px 0;
}}

QLabel#homeSubtitle {{
    font-size: 15px;
    color: {TEXT_SECONDARY};
    padding-bottom: 32px;
}}

/* 卡片容器 */
QFrame#card {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 24px;
}}
QFrame#card:hover {{
    border: 1px solid {PRIMARY};
}}

QLabel#cardIcon {{
    font-size: 42px;
    padding: 8px 0;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    padding: 8px 0;
}}

QLabel#cardDesc {{
    color: {TEXT_SECONDARY};
    font-size: 13px;
    padding: 4px 0;
}}

/* 主按钮 */
QPushButton#primaryButton {{
    background: {PRIMARY};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: 600;
    min-height: 28px;
}}
QPushButton#primaryButton:hover {{
    background: {PRIMARY_HOVER};
}}
QPushButton#primaryButton:pressed {{
    background: {PRIMARY_HOVER};
}}
QPushButton#primaryButton:disabled {{
    background: {BORDER};
    color: {TEXT_SECONDARY};
}}

/* 次级按钮 */
QPushButton#secondaryButton {{
    background: {BG};
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: 500;
    min-height: 24px;
}}
QPushButton#secondaryButton:hover {{
    background: {PRIMARY_LIGHT};
}}

/* 分区标题 */
QLabel#sectionTitle {{
    font-size: 20px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    padding: 16px 0 4px 0;
}}

QLabel#sectionDesc {{
    color: {TEXT_SECONDARY};
    padding-bottom: 16px;
}}

/* 分组框 */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {PRIMARY};
    background: {BG};
}}

/* 表格 */
QTableWidget {{
    border: 1px solid {BORDER};
    gridline-color: {BORDER};
    background: {BG};
    alternate-background-color: {BG_ALT};
}}
QHeaderView::section {{
    background: {BG_ALT};
    color: {TEXT_SECONDARY};
    padding: 6px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}

/* 输入控件 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 8px;
    background: {BG};
    min-height: 22px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {PRIMARY};
}}

QSlider::groove:horizontal {{
    border: 1px solid {BORDER};
    height: 4px;
    background: {BG_ALT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PRIMARY};
    border: none;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QRadioButton, QCheckBox {{
    color: {TEXT_PRIMARY};
    padding: 3px;
}}

/* 底栏 */
QFrame#bottomBar {{
    background: {BG_ALT};
    border-top: 1px solid {BORDER};
    min-height: 32px;
    max-height: 32px;
}}
QLabel#bottomHint {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 0 12px;
}}

/* 图像/结果显示区 */
QLabel#imageDisplay {{
    background: {BG_ALT};
    border: 1px dashed {BORDER};
    color: {TEXT_SECONDARY};
    qproperty-alignment: AlignCenter;
    min-height: 260px;
}}

/* 结果强调 */
QLabel#resultBig {{
    font-size: 32px;
    font-weight: 700;
    color: {PRIMARY};
}}
QLabel#resultLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
"""
