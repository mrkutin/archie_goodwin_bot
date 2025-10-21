from typing import List

from app.tools.factory import create_code_tools

# Build tools for each code: (collection, display name, allow_fractional)
_CODE_SPECS = [
    ("APK-RF", "Арбитражный процессуальный кодекс РФ", False),
    ("BK-RF", "Бюджетный кодекс РФ", False),
]

ALL_TOOLS: List[object] = []

for collection, name, allow_fractional in _CODE_SPECS:
    search_tool, exact_tool = create_code_tools(
        collection_name=collection,
        code_key=collection,
        full_display_name=name,
        allow_fractional_articles=allow_fractional,
    )
    # Preserve ordering: exact then search for each code
    ALL_TOOLS.append(exact_tool)
    ALL_TOOLS.append(search_tool)
