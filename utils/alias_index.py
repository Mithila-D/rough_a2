import json


class AliasIndex:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)

    def lookup(self, text: str):
        text = text.lower()
        matches = []

        for item in self.data:
            for alias in item.get("aliases", []):
                if alias.lower() in text:
                    matches.append(item)
                    break

        return matches