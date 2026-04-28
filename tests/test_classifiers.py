from src.domain.classifiers import classify_product


def test_classify_maiz():
    rules = {"Maíz": ["maíz", "maiz", "corn"]}
    assert classify_product("Los demás maíz", rules) == "Maíz"
    assert classify_product("Maiz amarillo", rules) == "Maíz"
    assert classify_product("Yellow corn", rules) == "Maíz"


def test_classify_otro():
    rules = {"Maíz": ["maíz"], "Trigo": ["trigo"]}
    assert classify_product("Cebada", rules) == "Otro"


def test_classify_empty():
    rules = {"Maíz": ["maíz"]}
    assert classify_product("", rules) == "Otro"
    assert classify_product(None, rules) == "Otro"
