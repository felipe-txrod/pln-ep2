from __future__ import annotations

import json

import pytest

from gpt_classifier.backend import BackendError
from gpt_classifier.batching import batch_request, parse_batch_response, prepare_batches
from gpt_classifier.codex_backend import CodexBackend
from gpt_classifier.data import Example, Text, canonical
from gpt_classifier.pipeline import run
from gpt_classifier.prompt import Config, build_request
from conftest import response


def config(**kwargs):
    return Config(backend="codex", batch_size=3, max_output_tokens=None, strategy="zero", per_class=0, **kwargs)


def request(text="alvo"):
    value, _ = build_request(Text(2,text),[],config())
    value["runtime"] = {"kind":"codex","endpoint":"codex.exec"}
    return value


def batch_response(items):
    result=response()
    result["output"][0]["content"][0]["text"]=json.dumps({"predictions":items})
    return result


def test_batch_deduplicates_examples_and_keeps_targets_separate():
    cfg=Config(backend="codex",max_output_tokens=None)
    examples=[Example(2,"exemplo um","c1"),Example(3,"exemplo dois","c234"),Example(4,"exemplo três","c5")]
    a,_=build_request(Text(100,"alvo A"),examples,cfg)
    b,_=build_request(Text(200,"alvo B"),examples,cfg)
    batch=batch_request([a,b])
    data=json.loads(batch["prompt"])
    assert len(data["examples"])==3
    assert [item["id"] for item in data["items"]]==["t0","t1"]
    assert all(set(item["target"])=={"resp_text"} for item in data["items"])
    assert data["items"][0]["example_ids"]==data["items"][1]["example_ids"]
    assert batch["schema"]["properties"]["predictions"]["minItems"]==2
    assert parse_batch_response(batch_response([{"id":"t1","label":"c5"},{"id":"t0","label":"c1"}]),batch["batch_ids"])=={"t0":"c1","t1":"c5"}


@pytest.mark.parametrize("items", [[], [{"id":"t0","label":"c1"}],
    [{"id":"t0","label":"c1"},{"id":"t0","label":"c5"}],
    [{"id":"t0","label":"c1"},{"id":"unknown","label":"c5"}],
    [{"id":"t0","label":"c6"},{"id":"t1","label":"c5"}],
    [{"id":"t0","label":"c1","reason":"x"},{"id":"t1","label":"c5"}]])
def test_batch_rejects_missing_duplicate_unknown_or_invalid_results(items):
    with pytest.raises(ValueError):
        parse_batch_response(batch_response(items),["t0","t1"])


def test_batch_size_and_byte_limits_split_without_truncating_targets():
    requests=[request("alvo "+str(i)+"x"*200) for i in range(5)]
    two=batch_request(requests[:2])
    limit=len(canonical(two).encode())+5
    batches=prepare_batches(requests,batch_size=3,max_bytes=limit)
    assert [len(mapping) for _,mapping in batches]==[2,2,1]
    assert all(len(canonical(payload).encode())<=limit for payload,_ in batches)
    assert sorted(i for _,mapping in batches for _,indices in mapping for i in indices)==list(range(5))


class FakeCodex(CodexBackend):
    def metadata(self):
        return {"kind":"codex","endpoint":"codex.exec","cli_version":"mock"}

    def _execute(self,args,prompt,directory):
        if getattr(self,"fail_at",None)==self.attempts:
            raise BackendError("simulated interruption")
        data=json.loads(prompt)
        labels={"A":"c1","B":"c234","C":"c5","D":"c1","E":"c234","F":"c5"}
        items=[{"id":item["id"],"label":labels[item["target"]["resp_text"]]} for item in reversed(data["items"])]
        (directory/"answer.json").write_text(json.dumps({"predictions":items}))
        (directory/"events.jsonl").write_text(json.dumps({"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20,"cached_input_tokens":0}})+"\n")


def test_batched_pipeline_preserves_rows_deduplication_and_usage(tmp_path):
    backend=FakeCodex(min_interval=0,max_requests=1)
    records=[Text(2,"A"),Text(3,"B"),Text(4,"A"),Text(5,"C")]
    kwargs=dict(backend=backend,output_dir=tmp_path/"run",cache_path=tmp_path/"cache.sqlite3")
    result=run(records,[],config(),**kwargs)
    assert [r["excel_row"] for r in result]==[2,3,4,5]
    assert [r["label"] for r in result]==["c1","c234","c1","c5"]
    assert backend.attempts==1 and result[2]["cache_hit"] is True
    manifest=json.loads((tmp_path/"run/run.json").read_text())
    assert manifest["unique_requests"]==1 and manifest["new_usage"]["input_tokens"]==100
    again=run(records,[],config(),**kwargs)
    assert all(r["cache_hit"] for r in again) and backend.attempts==1


def test_batched_pipeline_resumes_only_missing_batches(tmp_path):
    records=[Text(i+2,t) for i,t in enumerate("ABCDEF")]
    backend=FakeCodex(min_interval=0)
    backend.fail_at=2
    kwargs=dict(output_dir=tmp_path/"run",cache_path=tmp_path/"cache.sqlite3")
    with pytest.raises(BackendError):
        run(records,[],config(),backend=backend,**kwargs)
    assert not (tmp_path/"run/predictions.jsonl").exists()
    resumed=FakeCodex(min_interval=0,max_requests=1)
    result=run(records,[],config(),backend=resumed,**kwargs)
    assert resumed.attempts==1 and len(result)==6
    assert all(r["cache_hit"] for r in result[:3])
    assert [r["label"] for r in result]==["c1","c234","c5"]*2


def test_api_and_invalid_batch_configs_are_rejected():
    for kwargs in ({"batch_size":2},{"batch_size":0},{"batch_size":9},{"prompt_profile":"unknown"}):
        with pytest.raises(ValueError):
            Config(**kwargs)
