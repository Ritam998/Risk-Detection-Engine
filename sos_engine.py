import json
import re
import os
import logging
from typing import Dict, List


class SOSEngine:
   

    SEVERITY_SCORES = {
        "CRITICAL": 1.0,
        "HIGH": 0.8,
        "MEDIUM": 0.5,
        "LOW": 0.3
    }

    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger("SOSEngine")
        self.logger.setLevel(logging.INFO)

        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "sos_config.json")

        self.patterns = self._load_patterns(config_path)

    def _load_patterns(self, path: str) -> Dict[str, List[re.Pattern]]:
        """
        Load and compile regex patterns from JSON config.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SOS config file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_patterns = json.load(f)

        compiled = {}

        for severity, phrases in raw_patterns.items():
            compiled[severity.upper()] = [
                re.compile(rf"\b{re.escape(p.lower())}\b", re.IGNORECASE)
                for p in phrases
            ]

        return compiled

    def _normalize(self, text: str) -> str:
        """
        Normalize text for safer matching.
        """
        return text.lower().strip()

    def detect(self, text: str) -> Dict:
        """
        Detect SOS keywords and return structured result.
        """

        if not text or not isinstance(text, str):
            return self._empty_result()

        normalized = self._normalize(text)

        detected_matches = []

        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            patterns = self.patterns.get(severity, [])

            for pattern in patterns:
                match = pattern.search(normalized)
                if match:
                    detected_matches.append({
                        "severity": severity,
                        "phrase": match.group(),
                        "score": self.SEVERITY_SCORES.get(severity, 0.0)
                    })

                    
                    if severity == "CRITICAL":
                        self.logger.warning(
                            f"CRITICAL SOS detected: {match.group()}"
                        )
                        return self._build_response(detected_matches)

        if detected_matches:
            self.logger.info("Non-critical SOS detected")
            return self._build_response(detected_matches)

        return self._empty_result()

    def _build_response(self, matches: List[Dict]) -> Dict:
        """
        Build structured detection response.
        """
        highest = max(matches, key=lambda x: x["score"])

        return {
            "sos_flag": True,
            "highest_severity": highest["severity"],
            "confidence_score": highest["score"],
            "matched_phrases": matches
        }

    def _empty_result(self) -> Dict:
        return {
            "sos_flag": False,
            "highest_severity": None,
            "confidence_score": 0.0,
            "matched_phrases": []
        }
