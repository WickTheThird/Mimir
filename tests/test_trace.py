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


def test_a_route_with_its_handler_on_the_same_line_does_not_borrow_the_next(tmp_path):
    root = tmp_path / "svc"; (root / "internal/http").mkdir(parents=True)
    (root / "internal/http/server.go").write_text(
        'package http\n\nfunc r() {\n\tmux.Handle(pat.Post("/v2/signup"), s.postSignup())\n\tmux.Handle(pat.Post("/v2/apps"), s.postApps())\n}\n')
    (root / "internal/http/h.go").write_text('package http\n\nfunc (s *Server) postSignup() {}\nfunc (s *Server) postApps() {}\n')
    t = trace(root, "signup")
    assert [r["handlers"] for r in t["routes"]] == [["postSignup"]]


def test_the_answer_is_ordered_cited_and_built_from_the_trace(tmp_path):
    from mimir.tools.trace import answer_markdown

    md = answer_markdown(trace(_repo(tmp_path), "embedded signup"))
    order = [md.index(h) for h in ("1. Entry points", "2. Handlers", "3. Background", "4. States")]
    assert order == sorted(order)
    assert "`POST /v2/whatsapp_signup/initiate` → `postInitSignup` (internal/http/server.go:4)" in md
    assert "`StoreSignupEvent` (internal/db/storage.go:3)" in md
    assert "started by `NewSignupFSM` in cmd/svc/main.go:4" in md
    assert "1. `SUBSCRIBE_WEBHOOKS`" in md and "2. `REGISTER_PHONE_NUMBER`" in md


def test_a_literal_is_followed_to_its_constant_its_check_and_its_route(tmp_path):
    from mimir.tools.trace import symbol_markdown, trace_symbol

    root = tmp_path / "svc"; (root / "internal/http").mkdir(parents=True)
    (root / "internal/http/server.go").write_text('package http\n\nfunc r() {\n\tmux.Handle(pat.Post("/v2/signup/initiate"),\n\t\ts.postInit())\n}\n')
    (root / "internal/http/signup.go").write_text(
        'package http\n\nconst (\n\tfinishEvent = "FINISH_ONBOARDING"\n)\n\n'
        'func signupType(e string) int {\n\tswitch e {\n\tcase finishEvent:\n\t\treturn 1\n\t}\n\treturn 0\n}\n\n'
        'func (s *Server) postInit() {\n\tt := signupType(payload.Event)\n\t_ = t\n}\n')
    r = trace_symbol(root, "FINISH_ONBOARDING")
    assert r["bindings"] == [{"name": "finishEvent", "path": "internal/http/signup.go", "line": 4}]
    (u,) = [u for u in r["uses"] if u["kind"] == "compared"]
    assert u["function"] == "signupType" and u["line"] == 9
    assert any(c["caller"] == "postInit" and "payload.Event" in c["text"] for c in r["callers"])
    assert any(rt["url"] == "/v2/signup/initiate" and rt["handler"] == "postInit" for rt in r["routes"])
    md = symbol_markdown(r)
    assert "**Where it is received and checked**" in md and "`POST /v2/signup/initiate` → `postInit`" in md


def test_code_tokens_are_what_the_operator_typed_as_code():
    from mimir.tools.trace import code_tokens

    assert code_tokens("where do we receive FINISH_WHATSAPP_BUSINESS_APP_ONBOARDING") == ["FINISH_WHATSAPP_BUSINESS_APP_ONBOARDING"]
    assert "signupTypeForEvent" in code_tokens("who calls signupTypeForEvent")
    assert "store_signup_event" in code_tokens("find store_signup_event")
    assert code_tokens("why is the api pod restarting") == []


def test_a_mock_that_panics_does_not_make_real_code_unimplemented(tmp_path):
    from mimir.tools.trace import gap_verdict

    root = tmp_path / "svc"; (root / "internal/facebook").mkdir(parents=True)
    (root / "internal/facebook/client.go").write_text(
        'package facebook\n\nfunc (c *Client) StartMigration() error {\n\treturn c.post("set_payment_method_migration_intent")\n}\n')
    (root / "internal/facebook/client_test.go").write_text(
        'package facebook\n\nfunc (m *mock) StartMigration() error { panic("set_payment_method_migration_intent not implemented") }\n')
    (v1, v2) = gap_verdict(root, ["set_payment_method_migration_intent", "pause_migration"])
    assert v1["status"] == "implemented" and v1["sites"][0]["function"] == "StartMigration" and v1["tests"] == 1
    assert v2["status"] == "absent"


def test_a_plan_mirrors_the_existing_route_storage_interface_doubles_and_schema(tmp_path):
    from mimir.tools.trace import plan_change, plan_markdown, repo_shape

    root = tmp_path / "svc"
    files = {
        "internal/http/server.go": 'package http\n\nfunc r() {\n\tmux.Handle(pat.Post("/private/v2/jobs"), s.postJobPrivate())\n\tmux.Handle(pat.Get("/private/v2/jobs/:id"), s.getJobPrivate())\n}\n',
        "internal/http/jobs.go": 'package http\n\nfunc (s *Server) postJobPrivate() {\n\twritePublicError(w)\n\ts.db.CreateJob(j)\n}\n\nfunc (s *Server) getJobPrivate() {\n\twritePublicError(w)\n\ts.db.FetchJob(id)\n}\n',
        "internal/db/interface.go": 'package db\n\ntype DBStorage interface {\n\tCreateJob(j Job) error\n\tFetchJob(id string) (Job, error)\n\tSaveJob(j Job) error\n}\n',
        "internal/db/storage.go": 'package db\n\nfunc (st *Storage) CreateJob(j Job) error { return nil }\nfunc (st *Storage) FetchJob(id string) (Job, error) { return Job{}, nil }\nfunc (st *Storage) SaveJob(j Job) error { return nil }\n',
        "internal/fsm/job_test.go": 'package fsm\n\nfunc (m *fakeDB) SaveJob(j Job) error { return nil }\n',
        "internal/fsm/job.go": 'package fsm\n\ntype JobCoordinator struct{}\n\nfunc (c *JobCoordinator) step(j Job) {\n\tswitch j.Status {\n\tcase "QUEUED":\n\t\tc.db.SaveJob(j)\n\tcase "DONE":\n\t}\n}\n',
        "migrations/000007_add_jobs.up.sql": "CREATE TABLE jobs (status TEXT CHECK (status IN ('QUEUED','DONE')));\nCREATE INDEX x ON jobs (id) WHERE status <> 'DONE';\n",
    }
    for rel, text in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
    t = trace(root, "cancel job")
    md = plan_markdown(plan_change(root, t, "cancel", repo_shape(root, ["job"])))
    assert "`POST /private/v2/jobs/:id/cancel` → `cancelJobPrivate`" in md
    assert "shaped like `getJobPrivate`" in md and "writePublicError" in md
    assert "add `CancelJob` to the interface" in md
    assert "internal/fsm/job_test.go" in md and "test double" in md
    assert "migrations/000008_allow_cancelled_status.up.sql" in md
    assert "partial indexes" in md
