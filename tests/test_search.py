from postpilot.tools.search import make_tavily_search


class FakeTavilyClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {"title": "A", "content": "alpha", "url": "https://a.example", "score": 0.9},
                {"title": "B", "content": "beta", "url": "https://b.example", "score": 0.8},
            ]
        }


def test_tavily_adapter_maps_results_and_passes_params():
    client = FakeTavilyClient()
    search = make_tavily_search(max_results=2, client=client)

    results = search("llm agents")

    assert results == [
        {"title": "A", "content": "alpha", "url": "https://a.example"},
        {"title": "B", "content": "beta", "url": "https://b.example"},
    ]
    assert client.calls == [
        {"query": "llm agents", "max_results": 2, "search_depth": "advanced"}
    ]


def test_tavily_adapter_handles_empty_response():
    class Empty:
        def search(self, **kwargs):
            return {}

    assert make_tavily_search(client=Empty())("q") == []
