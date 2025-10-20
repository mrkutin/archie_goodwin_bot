from typing import List

from app.tools.factory import create_code_tools

# Build tools for each code: (collection, display name, allow_fractional)
_CODE_SPECS = [
    ("APK-RF", "Арбитражный процессуальный кодекс РФ", False),
    # ("uk_1996", "Уголовный кодекс РФ", False),
    # ("gk_1994", "Гражданский кодекс РФ", False),
    # ("apk_2002", "Арбитражный процессуальный кодекс РФ", False),
    # ("gpk_2002", "Гражданский процессуальный кодекс РФ", False),
    # ("gsk_2004", "Градостроительный кодекс РФ", False),
    # ("koap_2001", "КоАП РФ", True),  # supports fractional article numbers
    # ("ktm_1999", "Кодекс торгового мореплавания РФ", False),
    # ("kvvt_2001", "Кодекс внутреннего водного транспорта РФ", False),
    # ("lk_2006", "Лесной кодекс РФ", False),
    # ("nk_1998", "Налоговый кодекс РФ", False),
    # ("sk_1995", "Семейный кодекс РФ", False),
    # ("tk_2001", "Трудовой кодекс РФ", False),
    # ("uik_1997", "Уголовно-исполнительный кодекс РФ", False),
    # ("upk_2001", "Уголовно-процессуальный кодекс РФ", False),
    # ("vk_2006", "Водный кодекс РФ", False),
    # ("vozk_1997", "Воздушный кодекс РФ", False),
    # ("zhk_2004", "Жилищный кодекс РФ", False),
    # ("zk_2001", "Земельный кодекс РФ", False),
    # ("bk_1998", "Бюджетный кодекс РФ", False),
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
