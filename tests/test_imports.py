from telegram_style_transfer.paths import CONFIG_DIR, CONFIGS_DIR, DATA_DIR, MODELS_DIR


def test_paths_are_constructed():
    assert CONFIG_DIR.name == "config"
    assert CONFIGS_DIR.name == "configs"
    assert DATA_DIR.name == "data"
    assert MODELS_DIR.name == "models"
