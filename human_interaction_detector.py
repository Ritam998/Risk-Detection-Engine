
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InteractionType(str, Enum):
    GENUINE_HUMAN   = "GENUINE_HUMAN"     
    CASUAL_VENTING  = "CASUAL_VENTING"    
    REHEARSED_INPUT = "REHEARSED_INPUT"   
    AMBIGUOUS       = "AMBIGUOUS"         


class EscalationLevel(str, Enum):
    NONE        = "NONE"         
    MONITOR     = "MONITOR"     
    SOFT_ALERT  = "SOFT_ALERT"  
    HARD_ALERT  = "HARD_ALERT"   
    SOS         = "SOS"         



class SOSKeywordEngine:
    

    # Tier 1 — Immediate SOS trigger regardless of any other score
    _TIER_1: frozenset[str] = frozenset({
        "want to kill myself", "going to kill myself", "end my life",
        "want to die", "planning to die", "i will suicide", "commit suicide",
        "kill myself tonight", "don't want to live", "no reason to live",
        "goodbye forever", "final goodbye", "won't be here tomorrow",
        "taking my life", "end it all", "can't go on",
    })

    # Tier 2 — Hard Alert, route to counsellor
    _TIER_2: frozenset[str] = frozenset({
        "i give up", "nobody cares", "completely alone", "no one would miss me",
        "better off without me", "can't take it anymore", "everything is hopeless",
        "nothing matters", "i'm done", "what's the point", "pointless to continue",
        "exhausted of everything", "can't keep going", "losing the will",
        "so tired of living", "feel like disappearing",
    })

    # Tier 3 — Soft Alert, supportive response
    _TIER_3: frozenset[str] = frozenset({
        "feeling empty", "so worthless", "hate myself", "i'm a burden",
        "nobody understands", "feel invisible", "completely lost",
        "don't care anymore", "stopped caring", "no motivation",
        "can't get out of bed", "crying all the time", "numb inside",
    })

    # Casual phrases that LOOK like distress but aren't (context negators)
    _CASUAL_NEGATORS: frozenset[str] = frozenset({
        "this exam is killing me", "dying of boredom", "dead tired",
        "killing it", "dead serious", "literally dying", "i'm dead 😂",
        "dying laughing", "this is killing me lol", "i could kill for",
    })

    def __init__(self) -> None:
        # Build hash maps for O(1) lookup
        self._tier1_map: dict[str, int] = {phrase: 1 for phrase in self._TIER_1}
        self._tier2_map: dict[str, int] = {phrase: 2 for phrase in self._TIER_2}
        self._tier3_map: dict[str, int] = {phrase: 3 for phrase in self._TIER_3}
        self._negator_hashes: set[str] = {
            self._hash(p) for p in self._CASUAL_NEGATORS
        }

    def scan(self, text: str) -> tuple[int, list[str]]:
        """
        Scan text for SOS keywords.

        Returns
        -------
        (tier, matched_phrases)
        tier=0 means no match. tier=1 is highest priority.
        """
        normalized = text.lower().strip()

        # Check if it's a known casual phrase first
        if self._is_casual(normalized):
            return 0, []

        matched: list[str] = []
        highest_tier = 0

        for phrase, tier in self._tier1_map.items():
            if phrase in normalized:
                matched.append(phrase)
                highest_tier = min(highest_tier or tier, tier)

        for phrase, tier in self._tier2_map.items():
            if phrase in normalized:
                matched.append(phrase)
                if highest_tier == 0 or tier < highest_tier:
                    highest_tier = tier

        for phrase, tier in self._tier3_map.items():
            if phrase in normalized:
                matched.append(phrase)
                if highest_tier == 0 or tier < highest_tier:
                    highest_tier = tier

        return highest_tier, matched

    def _is_casual(self, text: str) -> bool:
        for negator in self._CASUAL_NEGATORS:
            if negator in text:
                return True
        return False

    @staticmethod
    def _hash(phrase: str) -> str:
        return hashlib.md5(phrase.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Human Pattern Detector
# ---------------------------------------------------------------------------

@dataclass
class MessageMeta:
    """Metadata about a single message (not its content)."""
    text_hash: str         # hash of message to detect copy-paste
    char_count: int
    word_count: int
    timestamp: float
    has_emoji: bool
    question_marks: int
    exclamation_marks: int


@dataclass
class HumanPatternWindow:
    """Sliding window of message metadata per user."""
    user_id: str
    metas: deque[MessageMeta] = field(default_factory=deque)
    last_activity_time: float = field(default_factory=time.time)


class HumanPatternDetector:
    """
    Analyses message metadata patterns (NOT content) to determine if
    the interaction feels like a genuine human in distress vs a bot,
    copy-paste test, or casual social use.

    Genuine distress signals:
      - Irregular message timing (not periodic/scripted)
      - Increasing message length over time (person elaborating)
      - Late-night / early-morning activity (3am–6am = elevated concern)
      - Short, fragmented messages (crisis = short bursts)
      - No repeated identical messages (genuine, not testing)
      - Emotional punctuation (??? !!! mixed)
    """

    _WINDOW_SIZE = 15
    _LATE_NIGHT_HOURS = range(22, 7)  # 10pm – 6am (wraps at midnight)

    def __init__(self, window_size: int = 15) -> None:
        self._window_size = window_size

    def analyse(
        self,
        window: HumanPatternWindow,
        text: str,
        timestamp: float,
    ) -> tuple[InteractionType, float, list[str]]:
        """
        Analyse interaction patterns and classify interaction type.

        Returns
        -------
        (interaction_type, authenticity_score 0.0–1.0, signal_list)
        """
        meta = self._extract_meta(text, timestamp)
        window.metas.append(meta)
        while len(window.metas) > self._window_size:
            window.metas.popleft()
        window.last_activity_time = timestamp

        signals: list[str] = []
        score = 0.5  # neutral start

        records = list(window.metas)

        # Signal 1: Repeated identical messages (bot / copy-paste)
        hashes = [m.text_hash for m in records]
        if len(records) >= 3 and len(set(hashes)) < len(hashes) * 0.6:
            score -= 0.3
            signals.append("REPEATED_IDENTICAL_MSGS: possible bot/test input")
        else:
            score += 0.1

        # Signal 2: Late night / early morning activity
        if self._is_late_night(timestamp):
            score += 0.2
            signals.append("LATE_NIGHT_ACTIVITY: high-risk time window (10pm–6am)")

        # Signal 3: Irregular timing (genuine human = irregular)
        if len(records) >= 4:
            intervals = [
                records[i].timestamp - records[i-1].timestamp
                for i in range(1, len(records))
            ]
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            if variance > 100:   # high variance = irregular = more human
                score += 0.1
                signals.append("IRREGULAR_TIMING: natural human interaction pattern")
            elif variance < 5:   # very uniform = possibly scripted
                score -= 0.15
                signals.append("UNIFORM_TIMING: robotic/scripted message cadence")

        # Signal 4: Fragmented short messages (crisis = short bursts)
        if len(records) >= 3:
            recent_words = [m.word_count for m in list(records)[-5:]]
            avg_words = sum(recent_words) / len(recent_words)
            if avg_words < 8:
                score += 0.15
                signals.append(f"SHORT_BURSTS: avg {avg_words:.1f} words (fragmented distress pattern)")

        # Signal 5: Emotional punctuation (???, !!!, mixed)
        total_q = sum(m.question_marks for m in records)
        total_e = sum(m.exclamation_marks for m in records)
        if total_q + total_e > len(records) * 1.5:
            score += 0.1
            signals.append(f"EMOTIONAL_PUNCTUATION: {total_q}? {total_e}! across messages")

        # Signal 6: No emojis in distress context (genuine crisis = plain text)
        recent_emojis = sum(1 for m in list(records)[-5:] if m.has_emoji)
        if recent_emojis == 0 and len(records) >= 3:
            score += 0.05
            signals.append("NO_EMOJIS: plain text pattern (consistent with genuine distress)")

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Classify
        if score >= 0.65:
            interaction_type = InteractionType.GENUINE_HUMAN
        elif score <= 0.3:
            interaction_type = InteractionType.REHEARSED_INPUT
        else:
            interaction_type = InteractionType.AMBIGUOUS

        return interaction_type, round(score, 4), signals

    def _extract_meta(self, text: str, timestamp: float) -> MessageMeta:
        text_hash = hashlib.md5(text.strip().lower().encode()).hexdigest()
        words = text.split()
        has_emoji = bool(re.search(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text
        ))
        return MessageMeta(
            text_hash=text_hash,
            char_count=len(text),
            word_count=len(words),
            timestamp=timestamp,
            has_emoji=has_emoji,
            question_marks=text.count('?'),
            exclamation_marks=text.count('!'),
        )

    def _is_late_night(self, timestamp: float) -> bool:
        import datetime
        hour = datetime.datetime.fromtimestamp(timestamp).hour
        return hour >= 22 or hour < 7


# ---------------------------------------------------------------------------
# Distress Authenticator
# ---------------------------------------------------------------------------

class DistressAuthenticator:
    """
    Distinguishes GENUINE crisis from casual/academic frustration.

    The key insight: students routinely use dramatic language about
    academic stress that sounds like crisis ("kill me", "dying",
    "want to give up") but is NOT a mental health emergency.

    This module uses contextual rules to separate:
      - Academic frustration  → "this exam is killing me, I hate this subject"
      - Genuine crisis        → "I can't do this anymore, I haven't slept in 4 days"

    Authenticity is determined by:
      1. Context markers (academic vs personal language)
      2. Temporal persistence (one-off vs repeated over days)
      3. Specificity of distress (vague casual vs detailed personal)
      4. Help-seeking vs help-avoiding language patterns
    """

    # Academic frustration markers — reduce crisis score if present
    _ACADEMIC_CONTEXT: frozenset[str] = frozenset({
        "exam", "test", "assignment", "homework", "deadline", "professor",
        "lecture", "subject", "marks", "grade", "project", "presentation",
        "semester", "class", "college", "university", "study", "studying",
        "viva", "paper", "internship", "placement",
    })

    # Personal distress markers — increase crisis authenticity if present
    _PERSONAL_DISTRESS: frozenset[str] = frozenset({
        "alone", "lonely", "nobody", "no one", "family", "parents",
        "relationship", "friend", "sleep", "eating", "crying", "tears",
        "nights", "weeks", "days", "months", "always", "never",
        "every day", "all the time", "for a long time",
    })

    # Help-seeking language (positive signal — person wants support)
    _HELP_SEEKING: frozenset[str] = frozenset({
        "help me", "need someone", "talk to someone", "please help",
        "is anyone there", "i need", "can you help", "tell me what to do",
        "don't know what to do", "who can i talk to",
    })

    # Help-avoiding / hopeless language (serious signal)
    _HELP_AVOIDING: frozenset[str] = frozenset({
        "no point", "won't help", "nothing will help", "useless to talk",
        "don't bother", "leave me alone", "don't care anymore", "too late",
        "past saving", "beyond help",
    })

    def authenticate(
        self,
        text: str,
        raw_score: float,
        sos_tier: int,
        session_day_count: int,
    ) -> tuple[float, list[str], bool]:
        """
        Compute authenticity multiplier and flags.

        Parameters
        ----------
        text               : message text
        raw_score          : NLP risk score 0–100
        sos_tier           : 0=none, 1=highest from SOSKeywordEngine
        session_day_count  : how many days this user has been active

        Returns
        -------
        (authenticity_multiplier 0.5–2.0, flags, is_academic_only)
        """
        lowered = text.lower()
        flags: list[str] = []
        multiplier = 1.0

        # Count context markers
        academic_hits = sum(1 for w in self._ACADEMIC_CONTEXT if w in lowered)
        personal_hits = sum(1 for w in self._PERSONAL_DISTRESS if w in lowered)
        help_seeking   = any(p in lowered for p in self._HELP_SEEKING)
        help_avoiding  = any(p in lowered for p in self._HELP_AVOIDING)

        is_academic_only = academic_hits > 0 and personal_hits == 0

        # Academic-only context → dampen crisis score
        if is_academic_only and sos_tier == 0:
            multiplier *= 0.6
            flags.append(f"ACADEMIC_CONTEXT: {academic_hits} academic markers, reducing crisis weight")

        # Personal distress context → amplify
        if personal_hits > 0:
            multiplier *= (1.0 + min(personal_hits * 0.15, 0.6))
            flags.append(f"PERSONAL_DISTRESS: {personal_hits} personal markers detected")

        # Help-seeking → slight positive (person wants support, good)
        if help_seeking:
            multiplier *= 1.1
            flags.append("HELP_SEEKING: user is reaching out for support")

        # Help-avoiding → serious signal
        if help_avoiding:
            multiplier *= 1.35
            flags.append("HELP_AVOIDING: user rejecting help — serious concern")

        # Persistence bonus — recurring distress across days
        if session_day_count >= 3:
            multiplier *= 1.2
            flags.append(f"PERSISTENT_DISTRESS: active {session_day_count} days")

        # SOS tier override — bypass academic dampening
        if sos_tier == 1:
            multiplier = max(multiplier, 2.0)
            flags.append("SOS_OVERRIDE: Tier-1 keyword bypasses academic dampening")
        elif sos_tier == 2:
            multiplier = max(multiplier, 1.6)
            flags.append("SOS_OVERRIDE: Tier-2 keyword — route to counsellor")

        return round(max(0.5, min(2.0, multiplier)), 4), flags, is_academic_only


# ---------------------------------------------------------------------------
# Auto Escalation Trigger
# ---------------------------------------------------------------------------

@dataclass
class EscalationDecision:
    """Final output of the auto-escalation pipeline."""
    level: EscalationLevel
    interaction_type: InteractionType
    authenticity_score: float
    sos_tier: int
    is_academic_only: bool
    auto_action: str          # what Rayza does automatically
    counsellor_notified: bool
    sos_activated: bool
    all_flags: list[str]
    reason: str
    timestamp: float = field(default_factory=time.time)


class AutoEscalationTrigger:
  

    def decide(
        self,
        final_score: float,
        sos_tier: int,
        interaction_type: InteractionType,
        authenticity_score: float,
        is_academic_only: bool,
        all_flags: list[str],
    ) -> EscalationDecision:

        level: EscalationLevel
        auto_action: str
        counsellor_notified = False
        sos_activated = False
        reason: str

        # ── Tier 1 SOS: immediate escalation regardless ────────────────
        if sos_tier == 1:
            level = EscalationLevel.SOS
            auto_action = "EMERGENCY: SOS triggered, emergency contacts notified, counsellor paged immediately"
            counsellor_notified = True
            sos_activated = True
            reason = "Tier-1 SOS keyword detected — immediate crisis"

        # ── Tier 2: hard alert ──────────────────────────────────────────
        elif sos_tier == 2 and authenticity_score >= 0.45:
            level = EscalationLevel.HARD_ALERT
            auto_action = "Counsellor notified, crisis support resources sent to user"
            counsellor_notified = True
            reason = f"Tier-2 crisis phrase with authenticity={authenticity_score:.2f}"

        # ── High composite score + genuine interaction ──────────────────
        elif final_score >= 75 and authenticity_score >= 0.65 and not is_academic_only:
            level = EscalationLevel.HARD_ALERT
            auto_action = "Counsellor notified, user offered immediate chat support"
            counsellor_notified = True
            reason = f"Score={final_score:.1f}, authenticity={authenticity_score:.2f} — genuine crisis"

        # ── Tier 3 or moderate score ────────────────────────────────────
        elif (sos_tier == 3 and authenticity_score >= 0.55) or \
             (final_score >= 55 and authenticity_score >= 0.5 and not is_academic_only):
            level = EscalationLevel.SOFT_ALERT
            auto_action = "Supportive message sent, mental health resources offered, flagged for follow-up"
            reason = f"Soft crisis signals — tier={sos_tier}, score={final_score:.1f}"

        # ── Academic-only context ──────────────────────────────────────
        elif is_academic_only and final_score >= 45:
            level = EscalationLevel.MONITOR
            auto_action = "Academic stress noted — monitor for escalation over next 24h"
            reason = "Academic frustration pattern — not crisis level"

        # ── No action ─────────────────────────────────────────────────
        else:
            level = EscalationLevel.NONE
            auto_action = "No action required — normal interaction"
            reason = f"Score={final_score:.1f}, below threshold or academic context"

        return EscalationDecision(
            level=level,
            interaction_type=interaction_type,
            authenticity_score=authenticity_score,
            sos_tier=sos_tier,
            is_academic_only=is_academic_only,
            auto_action=auto_action,
            counsellor_notified=counsellor_notified,
            sos_activated=sos_activated,
            all_flags=all_flags,
            reason=reason,
        )
