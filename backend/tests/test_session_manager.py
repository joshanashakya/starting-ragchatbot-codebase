from session_manager import SessionManager


def test_sessions_get_sequential_ids():
    sm = SessionManager()
    assert sm.create_session() == "session_1"
    assert sm.create_session() == "session_2"


def test_history_is_none_for_unknown_empty_or_missing_session():
    sm = SessionManager()
    sid = sm.create_session()
    assert sm.get_conversation_history(None) is None
    assert sm.get_conversation_history("nope") is None
    assert sm.get_conversation_history(sid) is None


def test_history_formatting():
    sm = SessionManager()
    sid = sm.create_session()
    sm.add_exchange(sid, "hi", "hello")
    assert sm.get_conversation_history(sid) == "User: hi\nAssistant: hello"


def test_history_keeps_only_last_max_history_exchanges():
    sm = SessionManager(max_history=2)
    sid = sm.create_session()
    for i in range(5):
        sm.add_exchange(sid, f"q{i}", f"a{i}")
    assert len(sm.sessions[sid]) == 4
    history = sm.get_conversation_history(sid)
    assert "q2" not in history and history.startswith("User: q3")


def test_add_message_creates_unknown_session():
    sm = SessionManager()
    sm.add_message("custom", "user", "x")
    assert sm.get_conversation_history("custom") == "User: x"


def test_clear_session():
    sm = SessionManager()
    sid = sm.create_session()
    sm.add_exchange(sid, "q", "a")
    sm.clear_session(sid)
    assert sm.get_conversation_history(sid) is None
    sm.clear_session("missing")  # no error
