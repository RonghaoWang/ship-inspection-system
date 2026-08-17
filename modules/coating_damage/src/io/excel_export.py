"""
Excel 数据记录器。
"""

from openpyxl import Workbook

from config import AreaMethod


class ExcelRecorder:
    """
    封装 openpyxl 的行式写入，支持 RANSAC / 重心法两种表头。
    """

    HEADERS_RANSAC = [
        "Region",
        "Standard_Area_cm2",
        "Source_Area_cm2",
        "Fixed_Area_cm2",
        "c",
    ]
    HEADERS_CENTER = [
        "Region",
        "Standard_Area_cm2",
        "Source_Area_cm2",
        "Fixed_Area_cm2",
        "CV2",
        "CV",
        "mean",
        "std",
    ]

    def __init__(self, area_method: AreaMethod = AreaMethod.DEPTH_CENTER):
        self.book = Workbook()
        self.sheet = self.book.active
        if area_method == AreaMethod.RANSAC:
            headers = self.HEADERS_RANSAC
        elif area_method == AreaMethod.DEPTH_CENTER:
            headers = self.HEADERS_CENTER
        else:
            raise ValueError(
                f"不支持的 area_method: {area_method}，请使用 '{AreaMethod.RANSAC.value}' 或 '{AreaMethod.DEPTH_CENTER.value}'"
            )
        for col, value in enumerate(headers, start=1):
            self.sheet.cell(row=1, column=col, value=value)

    def append_row(self, data: list) -> None:
        """追加一行数据。"""
        self.sheet.append(data)

    def save(self, path: str) -> None:
        """保存到磁盘。"""
        self.book.save(path)
        print(f"[Excel] 已保存到 {path}")
