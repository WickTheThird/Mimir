"""trace_feature: route, handler, calls, worker, wiring, states, from code alone."""

import pathlib

from mimir.tools.trace import choose_term, key_terms, render, trace


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "svc"
    files = {
        "cmd/svc/main.go": 'package main\n\nfunc main() {\n\tw := fsm.NewSignupFSM(db)\n\tw.Start()\n}\n',
        "internal/http/server.go": 'package http\n\nfunc (s *Server) routes() {\n\tmux.Handle(pat.Post("/v2/whatsapp_signup/initiate"),\n\t\ts.postInitSignup())\n}\n',
        "internal/http/signup.go": 'package http\n\n// embedded signup entry point\nfunc (s *Server) postInitSignup() http.HandlerFunc {\n\ttok, _ := s.fb.ExchangeTokenCode(code)\n\ts.db.StoreSignupEvent(tok)\n\treturn nil\n}\n',
        "internal/db/storage.go": 'package db\n\nfunc (st *Storage) StoreSignupEvent(t string) {}\n\nfunc (st *Storage) WorkOnSignupEvent(h func()) {}\n',
        "internal/fb/client.go": 'package fb\n\nfunc (c *Client) ExchangeTokenCode(code string) (string, error) { return "", nil }\n',
        "internal/fsm/signup.go": 'package fsm\n\ntype SignupFSM struct{}\n\nfunc NewSignupFSM(db any) *SignupFSM { return nil }\n\nfunc (f *SignupFSM) machine(e any) {\n\tswitch e {\n\tcase "SUBSCRIBE_WEBHOOKS":\n\tcase "REGISTER_PHONE_NUMBER":\n\t}\n}\n',
        "vendor/github.com/x/y/signup.go": 'package y\n\nfunc VendoredSignupThing() {}\n',
    }
    for rel, text in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    return root


def test_key_terms_go_from_specific_to_single_words():
    assert key_terms("embedded signup")[:3] == ["embeddedsignup", "embedded_signup", "embeddedSignup"]
    assert "signup" in key_terms("embedded signup")


def test_the_term_is_the_one_code_actually_uses(tmp_path):
    term, _ = choose_term(_repo(tmp_path), "embedded signup", tests=False)
    assert term == "signup"


def test_a_full_trace_from_route_to_states(tmp_path):
    t = trace(_repo(tmp_path), "embedded signup")
    assert t["routes"][0]["handlers"] == ["postInitSignup"]          # handler on the next line
    (h,) = t["handlers"]
    assert {"ExchangeTokenCode", "StoreSignupEvent"} <= set(h["calls"])
    called = {c["name"]: c["path"] for c in t["calls"]}
    assert called["StoreSignupEvent"] == "internal/db/storage.go" and called["ExchangeTokenCode"] == "internal/fb/client.go"
    consumers = {c["name"] for c in t["consumers"]}
    assert {"WorkOnSignupEvent", "SignupFSM", "NewSignupFSM"} <= consumers
    assert any(w["path"] == "cmd/svc/main.go" for w in t["wiring"])
    assert [s["state"] for s in t["states"]] == ["SUBSCRIBE_WEBHOOKS", "REGISTER_PHONE_NUMBER"]
    assert "VendoredSignupThing" not in render(t)


def test_nothing_named_like_it_says_so(tmp_path):
    t = trace(_repo(tmp_path), "quantum teleport")
    assert render(t).startswith("no code names anything like")
