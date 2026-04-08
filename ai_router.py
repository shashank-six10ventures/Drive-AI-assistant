from __future__ import annotations

import json
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import settings


class AIRouter:
    def __init__(self, provider: Optional[str] = None, enabled: bool = True):
        self.provider = (provider or settings.ai_provider).lower()
        self.enabled = enabled

    def can_use_llm(self) -> bool:
        if not self.enabled:
            return False
        if self.provider in ["anthropic", "claude"]:
            return bool(settings.anthropic_api_key)
        if self.provider == "openai":
            return bool(settings.openai_api_key)
        if self.provider == "groq":
            return bool(settings.groq_api_key)
        if self.provider == "gemini":
            return bool(settings.gemini_api_key)
        if self.provider == "huggingface":
            return bool(settings.huggingface_api_key)
        return False

    def generate(self, prompt: str, system_prompt: str = "You are a helpful data analyst AI.") -> str:
        if not self.enabled:
            return ""
        if self.provider in ["anthropic", "claude"]:
            return self._anthropic(prompt, system_prompt)
        if self.provider == "openai":
            return self._openai(prompt, system_prompt)
        if self.provider == "groq":
            return self._groq(prompt, system_prompt)
        if self.provider == "gemini":
            return self._gemini(prompt, system_prompt)
        if self.provider == "huggingface":
            return self._huggingface(prompt)
        return "AI provider not configured. Falling back to rule-based output."

    def parse_query_intent(self, query: str) -> dict:
        if not self.can_use_llm():
            return self._heuristic_intent(query)
        prompt = (
            "Extract user intent as strict JSON with keys: "
            "intent, keywords, filters, action, metric, timeframe, folder_name, target_name. "
            "filters must include uploader, year, file_type, item_kind. "
            f"Query: {query}"
        )
        raw = self.generate(prompt, system_prompt="You are an intent parser. Return only valid JSON.")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return self._heuristic_intent(query)

    def search_web(self, query: str, domains: Optional[list[str]] = None) -> list[dict]:
        if not settings.tavily_api_key:
            return []
        try:
            payload = {
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            }
            if domains:
                payload["include_domains"] = domains
            response = requests.post("https://api.tavily.com/search", json=payload, timeout=45)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception:
            return []

    def search_amazon(self, query: str, marketplace: Optional[str] = None) -> list[dict]:
        domain = marketplace or settings.amazon_marketplace or "amazon.in"
        search_url = f"https://www.{domain}/s"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            response = requests.get(search_url, params={"k": query}, headers=headers, timeout=45)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for block in soup.select("[data-component-type='s-search-result']")[:8]:
                title_el = block.select_one("h2 span")
                link_el = block.select_one("h2 a")
                price_whole = block.select_one(".a-price-whole")
                price_fraction = block.select_one(".a-price-fraction")
                rating_el = block.select_one(".a-icon-alt")
                if not title_el or not link_el:
                    continue
                price = ""
                if price_whole:
                    price = price_whole.get_text(strip=True)
                    if price_fraction:
                        price = f"{price}.{price_fraction.get_text(strip=True)}"
                link = link_el.get("href", "")
                if link.startswith("/"):
                    link = f"https://www.{domain}{link}"
                results.append(
                    {
                        "title": title_el.get_text(strip=True),
                        "url": link,
                        "price": price,
                        "rating": rating_el.get_text(strip=True) if rating_el else "",
                        "source": domain,
                    }
                )
            return results
        except Exception:
            return []

    def _heuristic_intent(self, query: str) -> dict:
        q = query.lower()
        keywords = [k for k in ["revenue", "profit", "sales", "marketing", "finance"] if k in q]
        year = None
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            year = m.group(1)
        uploader = None
        u = re.search(r"uploaded by\s+([a-z0-9 _-]+)", q)
        if u:
            uploader = u.group(1).strip(" .,!?:;")
        file_type = None
        for t in ["pdf", "csv", "excel", "ppt", "doc"]:
            if t in q:
                file_type = t
        folder_name = None
        folder_match = re.search(r"(?:in|inside|under)\s+folder\s+([a-z0-9 _./-]+)", q)
        if not folder_match:
            folder_match = re.search(r"folder\s+([a-z0-9 _./-]+)", q)
        if folder_match:
            folder_name = folder_match.group(1).strip(" .,!?:;")
        target_name = None
        target_match = re.search(
            r"(?:download|rename|open|analyze|compare)\s+(?:file|folder)?\s*([a-z0-9 _./()-]+)",
            q,
        )
        if target_match:
            target_name = target_match.group(1).strip(" .,!?:;")
        action = "search"
        item_kind = None
        if "compare" in q:
            action = "compare"
        if "combine" in q or "merge" in q or "joined dataset" in q:
            action = "combine"
        if "trend" in q or "monthly" in q:
            action = "trend"
        if "dashboard" in q:
            action = "dashboard"
        if "brief" in q or "executive summary" in q or "weekly summary" in q:
            action = "executive_brief"
        if "alert" in q or "risk" in q or "what changed" in q:
            action = "alerts"
        if "download" in q or "export" in q:
            action = "download"
        if "compare with web" in q or "search web" in q:
            action = "web_compare"
        if "amazon" in q or "market price" in q or "marketplace" in q:
            action = "amazon_compare"
        if "clean naming" in q or "clean names" in q or "rename files" in q or "fix naming" in q:
            action = "rename_preview"
        if ("apply rename" in q or "apply naming cleanup" in q or "rename them" in q) and "rename" in q:
            action = "rename_apply"
        if "show contents" in q or "inside folder" in q or "in folder" in q:
            action = "folder_contents"
        if q.startswith("from these"):
            action = "follow_up"
        if "folder" in q or "folders" in q:
            item_kind = "folder"
        elif "file" in q or "files" in q or "dataset" in q or "datasets" in q:
            item_kind = "file"
        return {
            "intent": "analytics" if action in ["compare", "trend"] else "search",
            "keywords": keywords,
            "filters": {"uploader": uploader, "year": year, "file_type": file_type, "item_kind": item_kind},
            "action": action,
            "metric": keywords[0] if keywords else None,
            "timeframe": "monthly" if "monthly" in q else None,
            "folder_name": folder_name,
            "target_name": target_name,
        }

    def _openai(self, prompt: str, system_prompt: str) -> str:
        if not settings.openai_api_key:
            return "OPENAI_API_KEY not set."
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": settings.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"OpenAI request failed: {exc}"

    def _anthropic(self, prompt: str, system_prompt: str) -> str:
        if not settings.anthropic_api_key:
            return "ANTHROPIC_API_KEY not set."
        try:
            url = "https://api.anthropic.com/v1/messages"
            payload = {
                "model": settings.anthropic_model,
                "max_tokens": 800,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            parts = data.get("content", [])
            text_parts = [part.get("text", "") for part in parts if part.get("type") == "text"]
            return "\n".join(text_parts).strip()
        except Exception as exc:
            return f"Anthropic request failed: {exc}"

    def _groq(self, prompt: str, system_prompt: str) -> str:
        if not settings.groq_api_key:
            return "GROQ_API_KEY not set."
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            headers = {
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"Groq request failed: {exc}"

    def _gemini(self, prompt: str, system_prompt: str) -> str:
        if not settings.gemini_api_key:
            return "GEMINI_API_KEY not set."
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
            )
            payload = {
                "contents": [
                    {"parts": [{"text": f"{system_prompt}\n\nUser:\n{prompt}"}]},
                ]
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "Gemini returned no content."
            parts = candidates[0].get("content", {}).get("parts", [])
            return "\n".join([p.get("text", "") for p in parts]).strip()
        except Exception as exc:
            return f"Gemini request failed: {exc}"

    def _huggingface(self, prompt: str) -> str:
        if not settings.huggingface_api_key:
            return "HUGGINGFACE_API_KEY not set."
        try:
            url = f"https://api-inference.huggingface.co/models/{settings.huggingface_model}"
            headers = {"Authorization": f"Bearer {settings.huggingface_api_key}"}
            payload = {"inputs": prompt}
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            return json.dumps(data, indent=2)
        except Exception as exc:
            return f"HuggingFace request failed: {exc}"
