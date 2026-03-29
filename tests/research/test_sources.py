"""Test research source adapters (unit tests, no API calls)."""
from jaeval.research.sources.arxiv import PaperResult
from jaeval.research.sources.github import RepoResult
from jaeval.research.sources.huggingface import HFModelResult
from jaeval.research.sources.web import WebResult


class TestDataclasses:
    def test_paper_citation(self):
        p = PaperResult(
            title="Test Paper",
            authors=["Author A", "Author B"],
            abstract="Abstract",
            arxiv_id="2301.00001",
            url="https://arxiv.org/abs/2301.00001",
            published="2023-01-01T00:00:00",
            categories=["cs.CL"],
        )
        assert "Author A" in p.citation
        assert "2023" in p.citation

    def test_repo_citation(self):
        r = RepoResult(
            name="test",
            full_name="user/test",
            description="desc",
            url="https://github.com/user/test",
            stars=100,
            language="Python",
            updated="2023-01-01",
            topics=[],
        )
        assert "100" in r.citation

    def test_hf_model_citation(self):
        m = HFModelResult(
            model_id="test/model",
            author="test",
            pipeline_tag="asr",
            downloads=1000,
            likes=50,
            url="https://huggingface.co/test/model",
        )
        assert "1000" in m.citation

    def test_web_result_citation(self):
        w = WebResult(
            title="Test Page",
            url="https://example.com/test",
            description="A test page description",
        )
        assert "Test Page" in w.citation
        assert "example.com" in w.citation
