import bpy

from . import zh_HANS

TRANSLATION_DOMAIN = "simple_camera_match"

LANGS = {
    "zh_HANS": zh_HANS.data,
}

TRANSLATIONS_DICT = {}


def build_translations_dict():
    translations_dict = {}
    for lang_code, data in LANGS.items():
        lang_dict = translations_dict.setdefault(lang_code, {})
        for src, src_trans in data.items():
            # ("Operator", src) 对应 operator 的 bl_label / docstring，
            # ("*", src) 是默认上下文（面板 label、属性 name/description 等）。
            # 不要再写 (TRANSLATION_DOMAIN, src)：translations_dict 的 key 第一项
            # 是"翻译上下文"而不是 domain，而插件里所有调用都是单参数的
            # pgettext_iface(msgid)，那个条目永远不会命中（纯冗余）。
            lang_dict[("Operator", src)] = src_trans
            lang_dict[("*", src)] = src_trans
    return translations_dict


def register():
    global TRANSLATIONS_DICT
    TRANSLATIONS_DICT = build_translations_dict()

    try:
        bpy.app.translations.unregister(TRANSLATION_DOMAIN)
    except ValueError:
        pass

    try:
        bpy.app.translations.register(TRANSLATION_DOMAIN, TRANSLATIONS_DICT)
    except Exception as e:
        print(f"[SimpleCameraMatch] Translation register error: {e}")


def unregister():
    global TRANSLATIONS_DICT

    try:
        bpy.app.translations.unregister(TRANSLATION_DOMAIN)
    except ValueError:
        pass

    TRANSLATIONS_DICT = {}
