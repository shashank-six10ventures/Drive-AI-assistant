from __future__ import annotations

import json
import re
from typing import Optional

import requests

from config import settings


class AIRouter:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.ai_provider).lower()

    def generate(self, prompt: str, system_prompt: str = "You are a helpful data analyst AI.") -> str:
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
        prompt = (
            "Extract user intent as strict JSON with keys: "
            "intent, keywords, filters, action, metric, timeframe. "
            "filters must include uploader, year, file_type. "
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
        action = "search"
        if "compare" in q:
            action = "compare"
        if "trend" in q or "monthly" in q:
            action = "trend"
        if q.startswith("from these"):
            action = "follow_up"
        return {
            "intent": "analytics" if action in ["compare", "trend"] else "search",
            "keywords": keywords,
            "filters": {"uploader": uploader, "year": year, "file_type": file_type},
            "action": action,
            "metric": keywords[0] if keywords else None,
            "timeframe": "monthly" if "monthly" in q else None,
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
