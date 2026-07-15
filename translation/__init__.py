import bpy

from . import zh_HANS

TRANSLATION_DOMAIN = "simple_camera_match"

LANGS = {
    "zh_CN": zh_HANS.data,
    "zh_HANS": zh_HANS.data,
}

TRANSLATIONS_DICT = {}


def build_translations_dict():
    translations_dict = {}
    for lang_code, data in LANGS.items():
        lang_dict = translations_dict.setdefault(lang_code, {})
        for src, src_trans in data.items():
            lang_dict[("Operator", src)] = src_trans
            lang_dict[("*", src)] = src_trans
            lang_dict[(TRANSLATION_DOMAIN, src)] = src_trans
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
