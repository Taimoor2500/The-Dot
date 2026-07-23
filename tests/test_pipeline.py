from google.genai.errors import APIError

from extraction.pipeline import _is_quota_exhausted


def _api_error(code: int, status: str) -> APIError:
    return APIError(code, {"error": {"code": code, "status": status, "message": "x"}})


def test_recognizes_quota_exhausted_error():
    assert _is_quota_exhausted(_api_error(429, "RESOURCE_EXHAUSTED")) is True


def test_other_429s_are_not_treated_as_quota_exhaustion():
    assert _is_quota_exhausted(_api_error(429, "SOME_OTHER_REASON")) is False


def test_non_429_errors_are_not_quota_exhaustion():
    assert _is_quota_exhausted(_api_error(500, "INTERNAL")) is False
