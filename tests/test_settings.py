from config.settings import Settings


def test_get_cors_list_single_origin():
    s = Settings(cors_origins="http://localhost:5173")
    assert s.get_cors_list() == ["http://localhost:5173"]


def test_get_cors_list_comma_separated():
    s = Settings(cors_origins="https://a.com,https://b.com")
    assert s.get_cors_list() == ["https://a.com", "https://b.com"]


def test_get_cors_list_json_array():
    s = Settings(cors_origins='["https://a.com"]')
    assert s.get_cors_list() == ["https://a.com"]
